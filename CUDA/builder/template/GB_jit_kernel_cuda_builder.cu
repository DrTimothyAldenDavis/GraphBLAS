//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_MEMORY (&Key, Key_size) ;       \
}

#define GB_FREE_ALL GB_FREE_WORKSPACE

//------------------------------------------------------------------------------
// typedefs and geometry
//------------------------------------------------------------------------------

// GB_key_t: sorting Key for CUB radix sort
typedef struct
{
    GB_KEY_TYPE i ;     // GB_KEY_TYPE is int32_t or int64_t
    #if GB_IS_MATRIX
    GB_KEY_TYPE j ;     // vectors have i only; matrices have i,j
    #endif
}
GB_key_t ;

#define CHUNKSIZE1          GB_CUDA_BUILDER_CHUNKSIZE1
#define LOG2_CHUNKSIZE1     GB_CUDA_BUILDER_CHUNKSIZE1_LOG2
#define BLOCKDIM1           GB_CUDA_BUILDER_BLOCKDIM1
#define ITEMS_PER_THREAD1   (CHUNKSIZE1 / BLOCKDIM1)

#define CHUNKSIZE2          GB_CUDA_BUILDER_CHUNKSIZE2
#define LOG2_CHUNKSIZE2     GB_CUDA_BUILDER_CHUNKSIZE2_LOG2
#define BLOCKDIM2           GB_CUDA_BUILDER_BLOCKDIM2
#define ITEMS_PER_THREAD2   (CHUNKSIZE2 / BLOCKDIM2)

// Int can be uint16_t if CHUNKSIZE1 and CHUNKSIZE2 are both < 65,535
#define Int uint16_t

//------------------------------------------------------------------------------
// GB_cuda_builder_phase1:
//------------------------------------------------------------------------------

// phase1 loads the (I,J,X) tuples into the (Key,Sx) workspace, and checks the
// indices (I,J) to ensure they are in range.  It sets the global (*ok) scalar
// to true if all indices are in range, or false if any index is out of range.
// The CPU builder also returns the first invalid indices for the error message
// returned to the user application; this kernel does do that.

// TODO: phase1 could also check if the (I,J) indices are already in sorted
// order.

__global__ void GB_cuda_builder_phase1
(
    // output
    GB_key_t *Key,      // size nvals+1: Key [-1...nvals-1]
    GB_Sx_TYPE *Sx,     // size nvals+1: Sx  [-1...nvals-1], NULL for iso case
    bool *ok,           // if true: (I,J) are valid; false: (I,J) out of range
    // input
    GB_I_TYPE *I,       // size nvals
    GB_J_TYPE *J,       // size nvals, NULL if C is a vector
    GB_Sx_TYPE *X,      // size nvals, NULL for iso case
    int64_t vlen,       // vector-length dimension of C (for I indices)
    int64_t vdim,       // vector-dim dimension of C (for J indices)
    int64_t nvals       // # of tuples in (I,J,X)
)
{

    //--------------------------------------------------------------------------
    // load the (I,J) tuples into the Key workspace
    //--------------------------------------------------------------------------

    bool my_ok = true ;

    for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                 p < nvals ;
                 p += blockDim.x * gridDim.x)
    {

        // get the indices
        GB_I_TYPE i = I [p] ;
        #if GB_IS_MATRIX
        GB_J_TYPE j = J [p] ;
        #endif

        // check if the indices are in range
        my_ok = my_ok
            #if GB_IS_MATRIX
            && (j >= 0 && j < vdim)
            #endif 
            && (i >= 0 && i < vlen) ;

        // load the indices into the Key workspace
        Key [p].i = (GB_KEY_TYPE) i ;
        #if GB_IS_MATRIX
        Key [p].j = (GB_KEY_TYPE) j ;
        #endif

        // load the values into the Sx workspace; no typecasting
        #if !GB_ISO_BUILD
        Sx [p] = X [p] ;
        #endif
    }

    //--------------------------------------------------------------------------
    // check if all indices are in range
    //--------------------------------------------------------------------------

    this_thread_block ( ).sync ( ) ;

    // TODO: reduce "my_ok" across the threadblock and then use an atomic AND
    // into global memory, into (*ok)

    this_thread_block ( ).sync ( ) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (theadIdx.x == 0 && blockIdx.x == 0)
    {
        // fillin the sentinel values for Key [-1].i, .j, and Sx [-1]
        Key [-1].i = (GB_KEY_TYPE) (-1) ;
        #if GB_IS_MATRIX
        Key [-1].j = (GB_KEY_TYPE) (-1) ;
        #endif
        #if !GB_ISO_BUILD
        memset (Sx [-1], 0, sizeof (GB_Sx_TYPE)) ;  // Sx [-1] = 0 (not used)
        #endif
    }
}

//------------------------------------------------------------------------------
// GB_cuda_builder_phase3
//------------------------------------------------------------------------------

// phase3 looks for duplicates in the sorted Key array.  It constructs a Map
// array which tells each entry in (Key,Sx) where it appears in C, after a
// cumulative sum.  It is nearly identical to CUDA/select phase1.

__global__ void GB_cuda_builder_phase3
(
    // outputs
    Int *Map,               // size nvals+1, in Map [-1...nvals-1]
    GB_Cp_TYPE *ChunkSum    // size nchunks_in_IJX+1,
                            // in ChunkSum [-1..nchunks_in_IJX]
    // inputs, not modified:
    GB_key_t *Key,          // size nvals+1: Key [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks_in_IJX
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock (IDENTICAL to select/phase1)
    //--------------------------------------------------------------------------

    __shared__ Int Local_Map [CHUNKSIZE1] ;

    // cub::Block* workspace:
    using BlockLoad  = cub::BlockLoad  <Int, BLOCKDIM1, ITEMS_PER_THREAD1> ;
    using BlockScan  = cub::BlockScan  <Int, BLOCKDIM1,
                                             cub::BLOCK_SCAN_WARP_SCANS> ;
    using BlockStore = cub::BlockStore <Int, BLOCKDIM1, ITEMS_PER_THREAD1> ;
    __shared__ union
    {
        typename BlockLoad::TempStorage load ;
        typename BlockScan::TempStorage scan ;
        typename BlockStore::TempStorage store ;
    } W ;

    //--------------------------------------------------------------------------
    // compute each local chunk of Map
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_IJX ;   // DIFFERS from select/phase1
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE1 ;
        int64_t my_chunk_size ;
        #if 0
        #else
        // this computation is just the 2nd #if case of select/phase1:
        int64_t plast = pfirst + CHUNKSIZE1 ;
        plast = GB_IMIN (plast, anz) ;
        my_chunk_size = plast - pfirst ;
        #endif

        //----------------------------------------------------------------------
        // determine the first unique tuple in each sequence of duplicates
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)       // block-stride loop
        {

            //------------------------------------------------------------------
            // this thread works on the p-th entry
            //------------------------------------------------------------------

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // determine if the p-th entry is kept
            //------------------------------------------------------------------

            // this phase DIFFERS from select/phase1

            // get the indices
            GB_KEY_TYPE iprev = Key [p-1].i ;
            GB_KEY_TYPE i     = Key [p  ].i ;
            #if GB_IS_MATRIX
            GB_KEY_TYPE jprev = Key [p-1].j ;
            GB_KEY_TYPE j     = Key [p  ].j ;
            #endif
            // keep = 1 if (i,j) is unique, 0 if duplicate
            bool keep = (i != iprev)
                #if GB_IS_MATRIX
                || (j != jprev)
                #endif
                ;

            //------------------------------------------------------------------
            // save the result in Local_Map, cumsum'd below
            //------------------------------------------------------------------

            Local_Map [pdelta] = keep ;
        }

        //----------------------------------------------------------------------
        // the remainder is IDENTICAL to select/phase1:
        //----------------------------------------------------------------------

        // clear the unused part of the Local_Map
        for ( ; pdelta < CHUNKSIZE1 ;
                pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+CHUNKSIZE1-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]).  This entire phase
        // computes the following::
        /*
            for (int i = 1 ; i < CHUNKSIZE1 ; i++)
            {
                Local_Map [i] += Local_Map [i-1] ;
            }
            Map [pfirst + 0:CHUNKSIZE1-1] = Local_Map [0:CHUNKSIZE1-1]
            block_aggregate = Local_Map [CHUNKSIZE1-1]
            ChunkSum [chunk] = block_aggregate ;
        */

        this_thread_block ( ).sync ( ) ;
        Int t [ITEMS_PER_THREAD1] ;

        // each thread loads its data from Local_Map (in shared memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD1 ; k++)
            {
                t [k] = Local_Map [ITEMS_PER_THREAD1 * threadIdx.x + k] ;
            }
        */
        BlockLoad (W.load).Load (Local_Map, t) ;
        this_thread_block ( ).sync ( ) ;

        // inclusive sum of data from Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i])
        Int block_aggregate ;
        BlockScan (W.scan).InclusiveSum (t, t, block_aggregate) ;
        this_thread_block ( ).sync ( ) ;

        // each thread saves its data into Map (in global memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD1 ; k++)
            {
                Map [pfirst + ITEMS_PER_THREAD1 * threadIdx.x + k] = t [k] ;
            }
        */
        BlockStore (W.store).Store (Map + pfirst, t) ;

        if (threadIdx.x == blockDim.x - 1)
        {
            // or try this:
//          ChunkSum [chunk] = tt [ITEMS_PER_THREAD1-1] ;   // in last thread
            ChunkSum [chunk] = block_aggregate ;
        }
    }

    //--------------------------------------------------------------------------
    // assign Map sentinel value
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Map [-1] = 0 ;  // sentinel value
    }
}

//------------------------------------------------------------------------------
// OUTLINE

/*  Example, assuming a chunksize of 4, with 14 tuples and 3 duplicates
    marked with (*).  The first entry of each set of duplicates is marked
    with (^)

    J:         [ 0 1 0 0 | 0 1 1 1 | 2 2 4 1 | 4 4 ]
    I:         [ 0 3 1 2 | 1 4 5 5 | 3 4 3 5 | 0 1 ]
    status:          ^     *   ^ *         *

    builder/phase1 loads the (I,J,X) tuples into (Key,Sx) and ensures the
    indices are in range.  builder/phase2 sorts them in (Key,Sx) has sorted
    tuples:

    output of builder/phase2 (Sx values not shown):
    Key.j:  -1 [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4 ]
    Key.i:  -1 [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3 ]
    status:        ^ *         ^ *   *

    builder/phase3 determines which entries are the first in a sequence of
    duplicates (output: Map), and which entries are the first in their
    respective vectors (output: JDelta).  The output after builder/phase3 is
    shown below, where LMap = Local_Map (a temporary shared array in each
    threadblock of phase3).  Map is the local cumsum of LMap.  Note the padding
    of LMap, Map, LJDelta, and JDelta.  Any given sequence of duplicates can
    span across the chunks (the (1,5) entry does so).

    LJDelta is 1 if the entry is the first in its vector (a leading entry), held
    in a temporary shared array in each threadblock.  It is computed from Key.j
    only and is not affected by the presence of duplicates.  JDelta is the local
    cumsum of LJDelta.  The (@) denotes the leading entry of each vector in C.

    inputs:
    Key.j:  -1 [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4 ]
    Key.i:  -1 [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3 ]
    status:        ^ *         ^ *   *
    output:
    LMap:      [ 1 1 0 1 | 1 1 1 0 | 0 1 1 1 | 1 1 - - ]
    Map:     0 [ 1 2 2 3 | 1 2 3 3 | 0 1 2 3 | 1 2 2 2 ]
    LJDelta: 0 [ 1 0 0 0 | 1 0 0 0 | 0 1 0 1 | 0 0 0 0 ]
    JDelta:  0 [ 1 1 1 1 | 1 1 1 1 | 0 1 1 2 | 0 0 0 0 ]
                 @         @           @   @           <---start of vectors in C

    ChunkSum [-1..nchunks_in_IXJ] is the # of non-duplicates to be kept in each
    chunk, with ChunkSum [-1] = 0 and ChunkSum [nchunks_in_IJX] = 0 for now.
    This matrix will have 11 total entries, which is the total sum of ChunkSum:
             0 [       3 |       3 |       3 |       2 ] 0

    JDeltaSum [-1..nchunks_in_IXJ] is the # of leading entries in each chunk,
    with JDeltaSum [-1] = 0.  This matrix has 4 unique values of J (0, 1, 2,
    and 4), which is the total sum of JDeltaSum:
             0 [       1 |       1 |       2 |       0 ] 0

    builder/phase4 allocates the output Cp, Ch, Ci, and Cx arrays and then
    moves the data from (Key,Sx) into (Cp,Ch,Ci,Cx), applying the dup operator
    to "sum" up the values of the duplicates as it does so.  Each sequence of
    duplicates is a handled by a single thread in a single threadblock.  This
    assumes there are not many duplicates for each entry.







        Need to do a global cumsum of First to know where to move the data

    Cj   [ 0 0 0 1 1 1 1 2 2 2 4 4 ..   ]
    Ci:  [ 0 1 2 3 4 5 7 3 4 5 0 1 ...
    Cx:  [ ...

    free S

    convert Cj into Cp and Ch:

    From Cj, compute Cj_delta (same as phase4 of select)

    Cj   [ 0 0 0 1 1 1 1 2 2 2 4 4 ..   ]
    Cjdel[ 0 0 0 1 0 0 0 1 0 0 1 0 ... ]
    Ci:  [ 0 1 2 3 4 5 7 3 4 5 0 1 ... ]
    Cx:  [ ... ]

    compute Cp and Ch from Cj_delta (like phase6 of select, but no Ah;
        use Cj [kC] instead of Ah [kA] in phase 6 of select)


    done!
*/

}

extern "C"
{
    GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_GET_CALLBACKS ;
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    GB_bix_alloc_f GB_bix_alloc = my_callback->GB_bix_alloc_func ;
    #endif

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    void *W_0 = NULL ; size_t W_0_size = 0 ;    // workspace of size nvals+1
    void *W_1 = NULL ; size_t W_1_size = 0 ;    // workspace of size nvals+1
    void *W_2 = NULL ; size_t W_2_size = 0 ;    // workspace of size nvals+1
    void *W_3 = NULL ; size_t W_3_size = 0 ;    // workspace size nchunks_max+2
    void *W_4 = NULL ; size_t W_4_size = 0 ;    // workspace size cnz+2

    // # of chunks
    int64_t cnz = 0 ;   // # of unique tuples, and # of entries in C
    int64_t nchunks_in_IJX = (nvals + CHUNKSIZE1 - 1) >> LOG2_CHUNKSIZE1 ;
    int64_t nchunks_in_C   = (nvals + CHUNKSIZE2 - 1) >> LOG2_CHUNKSIZE2 ;
    int64_t nchunks_max    = GB_IMAX (nchunks_in_IJX, nchunks_in_C) ;

    dim3 grid (gridsz) ;        // = min (ceil (nvals/CHUNKSIZE1), 256*(#sms))
    dim3 block1 (BLOCKDIM1) ;
    dim3 block2 (BLOCKDIM2) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    W_0 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_key_t), &W_0_size) ;
    W_1 = GB_MALLOC_MEMORY (nvals+1, sizeof (Int), &W_1_size) ;
    #if !GB_ISO_BUILD
    W_2 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_Sx_type), &W_2_size) ;
    #endif
    W_3 = GB_MALLOC_MEMORY (nchunks_max+2, sizeof (GB_Cp_TYPE), &W_3_size) ;
    if (W_0 == NULL || W_1 == NULL || W_3 == NULL
        #if !GB_ISO_BUILD
        || W_2 == NULL
        #endif
        )
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one so Key [-1...nvals-1], Sx [-1..nvals-1] and Map
    // [-1...nvals-1] can be used
    GB_key_t *Key = ((GB_key_t *) W_0) + 1 ;
    GB_Sx_TYPE *Sx = ((GB_Sx_TYPE *) W_2) + 1 ;
    Int *Map = ((Int *) W_1) + 1 ;
    // shift by one so ChunkSum [-1...nchunks_max] can be used
    GB_Cp_TYPE *ChunkSum = ((GB_Cp_TYPE *) W_3) + 1 ;

    //--------------------------------------------------------------------------
    // phase1: load the (Key,Sx) workspace and check if indices are in range
    //--------------------------------------------------------------------------

    // TODO: check for valid indices could be skipped if this method knows its
    // I,J inputs are already valid.

    bool ok = true ;

    GB_cuda_builder_phase1 <<<grid, block1, 0, stream>>>
        (Key, Sx, &ok, I, J, X, (int64_t) C->vlen, (int64_t) C->vdim, nvals) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    GB_OK (ok ? GrB_SUCCESS : GrB_INVALID_INDEX) ;

    //--------------------------------------------------------------------------
    // phase2: CUB radix sort of (Key,Sx)
    //--------------------------------------------------------------------------

    // TODO: could be skipped if phase1 detects (I,J) are already sorted in
    // phase1, or if it knows (I,J) are already sorted on input.

    //--------------------------------------------------------------------------
    // phase3: look for duplicates (compare with phase1 of CUDA/select)
    //--------------------------------------------------------------------------

    GB_cuda_builder_phase3 <<<grid, block1, 0, stream>>>
        ( /* outputs: */ Map, ChunkSum,
          /* inputs: */ Key, nvals, nchunks_in_IJX) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase4: sum up the unique entries in each chunk (on the CPU)
    //--------------------------------------------------------------------------

    // compare with phase2 of CUDA/select

    // At the start of phase4, ChunkSum [-1..nchunks_in_IJX] holds the # of
    // unique entries in C from each chunk of (I,J,X), with cnz = 11 unique
    // entries in C for this example:

    //       x [       3       1       2       3       2]
    // This phase computes an exclusive cumulative sum:
    //       0 [       0       3       4       6       9]    11

    ChunkSum [-1] = 0 ;         // sentinel value

    // overwrite ChunkSum [0..gridsz] with its cumulative sum
    for (int64_t chunk = 0 ; chunk < nchunks_in_IJX ; chunk++)
    {
        // get the # of entries found by this threadblock
        int64_t s = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = cnz ;
        cnz += s ;
    }
    ChunkSum [nchunks_in_IJX] = cnz ;

    //--------------------------------------------------------------------------
    // phase5: allocate C and construct Ci, Cx, and Ck1
    //--------------------------------------------------------------------------

    // compare with phase3 of CUDA/select

    // allocate the C matrix as hypersparse, with cnz entries
    GB_OK (GB_bix_alloc (C, cnz, GxB_HYPERSPARSE, false, true, GB_ISO_BUILD)) ;
    C->nvals = cnz ;
    if (cnz == 0)
    {
        // C is empty; nothing more to do
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    C->jumbled = false ; 

    // allocate workspace of size cnz+2
    W_4 = GB_MALLOC_MEMORY (cnz+2, sizeof (GB_Cj_SIGNED_TYPE), &W_4_size) ;
    if (W_4 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // use W_4 as workspace for Ck0 and Ck1, which are the same array, but Ck0
    // is accessed with 0-based indices (in range 0:cnz-1) and Ck1 uses 1-based
    // indices (in range 1:cnz).  That is, Ck0 [0] is the same as Ck1 [1].  The
    // first entry (Ck1 [0] and Ck0 [-1]) is a sentinel value (-1), set in the
    // phase3 kernel launch below.
    GB_Cj_SIGNED_TYPE *Ck0 = ((GB_Aj_SIGNED_TYPE *) W_4) + 1 ;
    GB_Cj_SIGNED_TYPE *Ck1 = ((GB_Aj_SIGNED_TYPE *) W_4) ;

    // The position where the pA-th entry in (I,J,X) appears in C is given by
    // pC = Map [pA] + ChunkSum [chunk], where pC is a 1-based index.  If this
    // position differs from the position of the (pA-1)st entry in (I,J,X),
    // then the entry is the first unique tuple in its set of duplicates.

    // Ck1 [pC] = kA = Ak [pA] if the entry in C is in the vector of C that
    // corresponds to the kA-th vector of A.

    // On output, where "|" reflects the chunks of A, not C, and spaces are
    // added in C just for illustration (denoting entries in A not kept):

    // input: FIXME: make an example for build; this is from select
    // Ai:     [ 0 1 2 2|5 6 0 1|8 1 7 2|7 8 9 3|7 9|- -]  size: anz = 18
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -] (- denotes empty)
    // Map:  0 [ 1 2 3 3|1 1 1 1|0 1 1 2|1 1 2 3|1 2 2 2] (note the padding)
    // Ak:     [ 0 0 0 1|1 1 2 3|3 4 4 5|5 5 5 6|6 6 - -]
    // ChunkSum:
    //       0 [       0       3       4       6       9]    11

    // output, with gaps denoting entries not in C:
    // Ci:     [ 0 1 2  |5      |  1   2|7   9 3|7 9 ]
    // Cx:     [ 1 1 1  |1      |  1   1|1   1 1|1 1 ]
    // Ck1: -1 [ 0 0 0  |1      |  4   5|5   5 6|6 6 ]
    //           ^       ^         ^   ^       ^---------start of vectors in C

    // Note that k=2 and k=3 in C are empty vectors since no entries in the 2nd
    // and 3rd columns of A were kept.

    GB_cuda_builder_phase5 <<<grid, block1, 0, stream>>>
        (/* outputs: */ C, Ck1,
         /* inputs: */  Key, Sx, ChunkSum, Map, nchunks) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Map (in W_1) no longer needed; reused below for Ck_Delta

    //--------------------------------------------------------------------------
    // phase 6: construct Ck_Delta and its local cumulative sum
    //--------------------------------------------------------------------------

    // compare with phase4 of CUDA/select

    // Ck_Delta [pC] = 1 if the pC-th entry is the first in its vector of C, or
    // 0 otherwise.  Then each threadblock computes the inclusive cumulative
    // sum of its chunk of Ck_Delta, overwriting Ck_Delta with its cumulative
    // sum.  Note the spaces (for illustration above) are removed here.
    // The index pC is 0-based in phase 6.

    // input:
    // Ck0: -1 [ 0 0 0 1|4 5 5 5|6 6 6 ]
    //           ^     ^ ^ ^     ^---------start of vectors in C
    // output:
    // Ck_Delta as 0/1:
    //       0 [ 1 0 0 1|1 1 0 0|1 0 0 ]
    // Ck_Delta as inclusive cumsum, per chunk of C:
    //       0 [ 1 1 1 2|1 2 2 2|1 1 1 ]
    // ChunkSum of C:
    //       0 [       2|      2|    1 ]

    // # of chunks in C:
    nchunks_in_C = (cnz + CHUNKSIZE2 - 1) >> LOG2_CHUNKSIZE2 ;

    // using W_1 [-1..cnz-1] as workspace for Ck_Delta, which is accessed
    // with 0-based indices, using Ck_Delta [-1..cnz-1] where cnz <= anz.
    // Note that W_1 [-1] is already set to zero in phase1 (Map [-1] = 0),
    // and thus Ck_Delta [-1] is already equal to 0, as required.
    Int *Ck_Delta = ((Int *) W_1) + 1 ;

    GB_cuda_builder_phase6 <<<grid, block2, 0, stream>>>
        (/* outputs: */ Ck_Delta, ChunkSum,
         /* inputs: */  Ck0, cnz, nchunks_in_C) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}
