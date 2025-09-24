//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_select_sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

using namespace cooperative_groups ;

#include "GB_cuda_ek_slice.cuh"

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_FREE_MEMORY (&W_0, W_0_size) ;   \
    GB_FREE_MEMORY (&W_1, W_1_size) ;   \
    GB_FREE_MEMORY (&W_2, W_2_size) ;   \
    GB_FREE_MEMORY (&W_3, W_3_size) ;   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE ;

#define CHUNK_SIZE 1024
#define LOG2_CHUNK_SIZE 10

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1: determine which entries in A to keep
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase1
(
    // outputs:
    GB_Aj_TYPE *Ak,     // size anz, values in range 0 to max # vectors in A
    GB_Ap_TYPE *Map,    // size anz, values in range 0 to anz-1
    int64_t *ChunkSum,  // size # threadblocks + 1
    // inputs, not modified:
    GrB_Matrix A,
    const void *ythunk,
    int64_t anz,            // # of entries in A
    int64_t nchunks_in_A    // # of chunks in A
)
{

    //--------------------------------------------------------------------------
    // get A and ythunk
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    #if ( GB_DEPENDS_ON_I )
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #endif
    #if ( GB_DEPENDS_ON_J) && ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if ( GB_DEPENDS_ON_Y )
    const GB_Y_TYPE y = * ((GB_Y_TYPE *) ythunk) ;
    #endif

    // workspace for each chunk
    __shared__ GB_Ap_TYPE Local_Map [CHUNK_SIZE] ;

    //--------------------------------------------------------------------------
    // compute Ak, and each local chunk of Map
    //--------------------------------------------------------------------------

    // Each threadblock (with blocksz = 512 threads) does an entire chunk (of
    // size CHUNK_SIZE = 1024).

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_A ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // determine the chunk and its slope
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNK_SIZE ;
        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, CHUNK_SIZE,
            &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        //----------------------------------------------------------------------
        // find the kth vector that contains each entry pA = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)
        {

            //------------------------------------------------------------------
            // determine the kth vector that contains the pA-th entry
            //------------------------------------------------------------------

            int64_t pA = pfirst + pdelta ;
            int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap,
                anvec1, kfirst, slope) ;

            //------------------------------------------------------------------
            // save the vector index kA, and determine if this entry is kept
            //------------------------------------------------------------------

            Ak [pA] = kA ;
            #if ( GB_DEPENDS_ON_J )
            int64_t j = GBh_A (Ah, kA) ;
            #endif
            #if ( GB_DEPENDS_ON_I )
            int64_t i = Ai [pA] ;
            #endif
            // keep = fselect (A (i,j)), 1 if A(i,j) is kept, else 0
            GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
            Local_Map [pdelta] = keep ;
        }

        // clear the unused part of the Local_Map
        for ( ; pdelta < CHUNK_SIZE ;
                pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
        }

        this_thread_block ( ).sync ( ) ;
        // FIXME: do a cub::BlockScan::InclusiveSum on threadblock's
        // Local_Map [0..CHUNK_SIZE-1],
        // where Local_Map [i] = sum (Local_Map [0:i]), so that
        // Local_Map [0] = 1 if the first entry is kept, 0 otherwise,
        // and Local_Map [CHUNK_SIZE-1] = total # entries kept in this block
        if (threadIdx.x == 0)
        {
            for (int i = 1 ; i < CHUNK_SIZE ; i++)
            {
                Local_Map [i] += Local_Map [i-1] ;
            }
        }
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // save the Local_Map in Map [pfirst..pfirst+CHUNK_SIZE-1]
        //----------------------------------------------------------------------

        for (pdelta = threadIdx.x ;
             pdelta < CHUNK_SIZE ;
             pdelta += blockDim.x)
        {
            Map [pfirst + pdelta] = Local_Map [pdelta] ;
        }

        //----------------------------------------------------------------------
        // save the # of entries kept in this chunk
        //----------------------------------------------------------------------

        if (threadIdx.x == blockDim.x - 1)
        {
            ChunkSum [chunk] = Local_Map [CHUNK_SIZE-1] ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase3: construct Ci, Cx, Ck
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase3
(
    // outputs:
    GrB_Matrix C,           // construct C->i and C->x
    GB_Aj_TYPE *Ck,         // size cnz+1
    // inputs, not modified:
    GrB_Matrix A,
    int64_t *ChunkSum,      // size # threadblocks + 1
    GB_Ap_TYPE *Map,        // size anz
    GB_Aj_TYPE *Ak,         // size anz
    int64_t anz,            // # of entries in A
    int64_t nchunks_in_A         // # of chunks in A
)
{

    //--------------------------------------------------------------------------
    // get C->i and C->x, shifting down by one to account for 1-based Map
    //--------------------------------------------------------------------------

    // in this method, the index pC = Map [pA] is 1-based, so decrement Ci, Cx,
    // to account for this.  Ck is already 1-based.

    GB_Ci_TYPE *Ci = C->i ; Ci-- ;
    #if !GB_ISO_SELECT
    GB_C_TYPE  *Cx = C->x ; Cx-- ;
    #endif

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    #if ( GB_DEPENDS_ON_I )
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #endif
    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    //--------------------------------------------------------------------------
    // select the entries and copy them into Ci, Cx; construct Ck
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_A ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the chunk of A; entries in pA = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNK_SIZE ;
        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, anz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // move the entries in this chunk of A into Ci and Cx
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            // get the position pC in C of the pA-th entry in A
            GB_Ap_TYPE pC = Map [pA  ] + ChunkSum [chunk] ;
            // get the position p0 in C of the (pA-1)-st entry in A
            GB_Ap_TYPE p0 = Map [pA-1] + ChunkSum [chunk - (pdelta == 0)] ;
            if (p0 < pC)
            {
                // This entry is kept; it appears in a new position pC as
                // compared to the entry Map [pA-1] immediately to its left.
                // Map contains 1-based indices since it was computed as an
                // inclusive cumsum, so Ci, Cx, and Ck have been shifted by one
                // above.
                Ci [pC] = Ai [pA] ;
                // Cx [pC] = Ax [pA] ;
                GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
                // save the index of the kA-th vector kA that holds this entry
                Ck [pC] = Ak [pA] ;
            }
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase4: construct Ck_Delta
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase4
(
    // outputs:
    GB_Cj_TYPE *Ck_Delta,   // size cnz
    int64_t *ChunkSum,      // size nchunks_in_C+2
    // inputs, not modified:
    GB_Aj_TYPE *Ck,         // size cnz+1
    int64_t cnz,
    int64_t nchunks_in_C
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ GB_Cj_TYPE Local_Ck_Delta [CHUNK_SIZE] ;

    //--------------------------------------------------------------------------
    // construct Ck_Delta and then cumsum each block
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_C ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the chunk of Ck and Ck_Delta this threadblock works on
        //----------------------------------------------------------------------

        // this threadblock works on Ck [pfirst:plast-1] and
        // Ck_Delta [pfirst:plast-1]

        int64_t pfirst = chunk << LOG2_CHUNK_SIZE ;
        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // determine which entries of Ck start new vectors in C
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)
        {
            int64_t pC = pfirst + pdelta ;
            Local_Ck_Delta [pdelta] = (Ck [pC-1] < Ck [pC]) ;
        }

        // clear the unused part of the Local_Ck_Delta
        for ( ; pdelta < CHUNK_SIZE ;
                pdelta += blockDim.x)
        {
            Local_Ck_Delta [pdelta] = 0 ;
        }

        this_thread_block ( ).sync ( ) ;
        // FIXME: do a cub::BlockScan::InclusiveSum on threadblock's
        // Local_Ck_Delta [0..CHUNK_SIZE-1]
        if (threadIdx.x == 0)
        {
            for (int i = 1 ; i < CHUNK_SIZE ; i++)
            {
                Local_Ck_Delta [i] += Local_Ck_Delta [i-1] ;
            }
        }
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // save the Local_Ck_Delta in Ck_Delta [pfirst:plast-1]
        //----------------------------------------------------------------------

        for (pdelta = threadIdx.x ;
             pdelta < CHUNK_SIZE ;
             pdelta += blockDim.x)
        {
            Ck_Delta [pfirst + pdelta] = Local_Ck_Delta [pdelta] ;
        }

        // last thread writes the sum of the whole threadblock to global
        if (threadIdx.x == blockDim.x - 1)
        {
            ChunkSum [chunk] = Local_Ck_Delta [CHUNK_SIZE-1] ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase6: construct Cp and Ch
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase6
(
    // outputs:
    GrB_Matrix C,           // Cp and Ch are constructed
    // inputs, not modified
    GB_Cj_TYPE *Ck_Delta,   // size cnz + 1
    int64_t *ChunkSum,      // size # threadblocks + 1
    GB_Aj_TYPE *Ck,         // size cnz + 1
    #if ( GB_A_IS_HYPER )
    GrB_Matrix A,
    #endif
    int64_t cnz,
    int64_t nchunks_in_C
)
{

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    // Cp and Ch use 1-based indexing below, so decrement them by 1
    GB_Cp_TYPE *Cp = C->p ; Cp-- ;
    GB_Cj_TYPE *Ch = C->h ; Ch-- ;
    #if ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif

    //--------------------------------------------------------------------------
    // determine the start of each vector in C
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_C ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get this chunk of C
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNK_SIZE ;
        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // compute Cp and Ch for this chunk
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pC = pfirst + pdelta ;
            // get the vector kC that contains the pC-th entry of C
            GB_Cj_TYPE kC = Ck_Delta [pC  ] + ChunkSum [chunk] ;
            // get the vector k0 that contains the (pC-1)-st entry of C
            GB_Cj_TYPE k0 = Ck_Delta [pC-1] + ChunkSum [chunk - (pdelta == 0)] ;
            if (k0 < kC)
            {
                // The pC-th entry  is the start of a vector kC in C;
                // note that kC is 1-based, so Cp and Ch are decremented above.
                int64_t kA = Ck [pC] ;
                Cp [kC] = pC ;
                Ch [kC] = GBh_A (Ah, kA) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // C->nvec is 0-based, so increment Cp to undo the Cp-- done above
        Cp++ ;
        Cp [C->nvec] = cnz ;
    }
}

//------------------------------------------------------------------------------
// select sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel)
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

    GrB_Info info ;

    // workspaces of size anz+2
    void *W_0 = NULL ; size_t W_0_size = 0 ;
    void *W_1 = NULL ; size_t W_1_size = 0 ;
    // workspace of size nchunks_in_A+1
    void *W_2 = NULL ; size_t W_2_size = 0 ;
    // workspace of size cnz+2, where cnz <= anz
    void *W_3 = NULL ; size_t W_3_size = 0 ;

    GB_A_NHELD (anz) ;          // # of entries in A
    int64_t cnz = 0 ;           // # of entries in C (which is <= anz)
    // # of chunks in A:
    int64_t nchunks_in_A = (anz + CHUNK_SIZE - 1) >> LOG2_CHUNK_SIZE ;

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;        // = min (ceil (nnz(A)/512), 256*(#sms))
    dim3 block (blocksz) ;      // = 512

    //--------------------------------------------------------------------------
    // phase 1: allocate workspace and determine which entries of A to keep
    //--------------------------------------------------------------------------

    // This phase constructs Ak [0..anz-1], where Ak [pA] = kA if the pA-th
    // entry is in the kA-th vector of A.  It also is the first phase in
    // constructing Map [-1..anz-1], where Map [pA] = pC + ChunkSum [chunk]
    // if the pA-th entry of A is the pC-th entry of C, and chunk is the
    // chunk of A that contains the pA-th entry.  The Map array is extended so
    // that it contains an integral number of chunks.

    // Example:  suppose the select operator fkeep(aij) keeps nonzero entries
    // in A, with the following input.  Suppose the CHUNK_SIZE is 4, for this
    // diagram (it is 1024 as #define'd above), as delineated by "|" in the
    // diagram.  The matrix A is 10 by 7 with 18 entries, held by column in
    // sparse form (Ah is NULL, and the kth vector is simply column k of A).

    // col:      0 0 0 1 1 1 2 3 3 4 4 5 5 5 5 6 6 6 (computed and put in Ak)
    // Ap:     [ 0     3     6 7   9  11      15     18 ]  size: ncol+1 = 8
    // Ai:     [ 0 1 2 2|5 6 0 1|8 1 7 2|7 8 9 3|7 9|- -]  size: anz = 18
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -]  size: anz = 18

    // phase1 computes Ak [0..anz-1] and Map [-1..anz-1], where Ak [pA] = kA if
    // the pA-th entry is in the kA-th vector of A, and where Map [pA] is an
    // inclusive cumulative sum of fkeep(aij) for each chunk.  In this example,
    // the first entry is kept, and this can be checked later by comparing Map
    // [-1] with Map [0].  Since A has entries equal to 0 or 1, fkeep(aij) is
    // simply aij, held in the Ax array:

    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -] (- denotes empty)
    // Map:  0 [ 1 2 3 3|1 1 1 1|0 1 1 2|1 1 2 3|1 2 2 2] (note the padding)
    // Ak:     [ 0 0 0 1|1 1 2 3|3 4 4 5|5 5 5 6|6 6 - -] (see col list above

    // ChunkSum [-1..nchunks_in_A] is the # of entries kept in each chunk, with
    // ChunkSum [-1] = 0 and ChunkSum [nchunks_in_A] = 0 for now:
    //       0 [       3       1       2       3       2] 0

    // The (Ai,Ak,Ax) sets of arrays are a coordinate-form representation of
    // the entries in A, in sorted order, with Ai being the row indices, Ak the
    // column indices (if the matrix is in CSC format) and Ax being the values.
    // If A is hypersparse by column, then Ak [pA] = kA holds the value kA if
    // the entry is in the kA-th nonempty column of A, which is column Ah [kA].

    size_t w0 = GB_IMAX (sizeof (GB_Aj_TYPE), sizeof (GB_Cj_TYPE)) ;
    W_0 = (void *) GB_MALLOC_MEMORY (anz+2, w0, &W_0_size) ;
    W_1 = (void *) GB_MALLOC_MEMORY (anz+2 + CHUNK_SIZE,
        sizeof (Gp_Ap_TYPE), &W_1_size) ;
    W_2 = (void *) GB_MALLOC_MEMORY (nchunks_in_A+2, sizeof (int64_t),
        &W_2_size) ;
    if (W_0 == NULL || W_1 == NULL || W_2 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // use W_0 [1..anz] as workspace for Ak [0..anz-1]
    GB_Aj_TYPE *Ak = (GB_Aj_TYPE *) W_0 ;

    // use W_1 workspace for Map, and shift by one to define Map [-1] as 0
    GB_Ap_TYPE *Map = ((GB_Ap_TYPE *) W_1) + 1 ;
    Map [-1] = 0 ;

    // ChunkSum [-1 .. nchunks_in_A-1] of size nchunks_in_A+1
    int64_t *ChunkSum = (int64_t *) W_2 - 1 ;
    ChunkSum [-1] = 0 ;

    // KERNEL LAUNCH 1: phase1
    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (/* outputs: */ Ak, Map, ChunkSum,
         /* inputs: */  A, ythunk, anz, nchunks_in_A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 2: sum up the entries in each block (on the CPU)
    //--------------------------------------------------------------------------

    // At the start of phase2, ChunkSum [-1..nchunks_in_A] holds the # of
    // entries kept in C from each chunk of A, with cnz = 11 entries kept in C.
    //       0 [       3       1       2       3       2]
    // This phase computes an exclusive cumulative sum:
    //       0 [       0       3       4       6       9]    11

    // overwrite ChunkSum [0..gridsdz] with its cumulative sum
    int64_t cnz = 0 ;
    for (int64_t chunk = 0 ; chunk < nchunks_in_A ; chunk++)
    {
        // get the # of entries found by this threadblock
        int64_t s = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = cnz ;
        cnz += s ;
    }
    ChunkSum [nchunks_in_A] = cnz ;

    //--------------------------------------------------------------------------
    // phase 3: allocate C and construct Ci, Cx, and Ck
    //--------------------------------------------------------------------------

    // allocate the C matrix as hypersparse, with cnz entries
    GB_OK (GB_bix_alloc (C, cnz, GxB_HYPERSPARSE, false, true, GB_ISO_SELECT)) ;
    C->nvals = cnz ;
    if (cnz == 0)
    {
        // C is empty; nothing more to do
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    // allocate workspace of size cnz+2
    W_3 = (void *) GB_MALLOC_MEMORY (cnz + 2, sizeof (GB_Aj_TYPE), &W_3_size) ;
    if (W_3 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // use W_3 as workspace for Ck, always accessed with 1-based indices
    GB_Aj_TYPE *Ck = ((GB_Aj_TYPE *) W_3) ;
    Ck [0] = -1 ;

    // The position where the pA-th entry in A appears in C is given by pC =
    // Map [pA] + ChunkSum [chunk], where pC is a 1-based index.  If this
    // position differs from the position of the (pA-1)st entry in A, then the
    // entry is kept, and is copied from A into C in phase3.

    // Ck [pC] = kA = Ak [pA] if the entry in C is in the vector of C that
    // corresponds to the kA-th vector of A.

    // On output, where "|" reflects the chunks of A, not C, and spaces are
    // added in C just for illustration (denoting entries in A not kept):

    // input:
    // Ai:     [ 0 1 2 2|5 6 0 1|8 1 7 2|7 8 9 3|7 9|- -]  size: anz = 18
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -] (- denotes empty)
    // Map:  0 [ 1 2 3 3|1 1 1 1|0 1 1 2|1 1 2 3|1 2 2 2] (note the padding)
    // Ak:     [ 0 0 0 1|1 1 2 3|3 4 4 5|5 5 5 6|6 6 - -]
    // ChunkSum:
    //       0 [       0       3       4       6       9]    11

    // output, with gaps denoting entries not in C:
    // Ci:     [ 0 1 2  |5      |  1   2|7   9 3|7 9 ]
    // Cx:     [ 1 1 1  |1      |  1   1|1   1 1|1 1 ]
    // Ck:  -1 [ 0 0 0  |1      |  4   5|5   5 6|6 6 ]
    //           ^       ^         ^   ^       ^---------start of vectors in C

    // Note that k=2 and k=3 in C are empty vectors since no entries in the 2nd
    // and 3rd columns of A were kept.

    // KERNEL LAUNCH 2: phase3
    GB_cuda_select_sparse_phase3 <<<grid, block, 0, stream>>>
        (/* outputs: */ C, Ck,
         /* inputs: */  A, ChunkSum, Map, Ak, anz, nchunks_in_A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ak (in W_0) no longer needed; reused below for Ck_Delta
    // Map (in W_1) no longer needed

    //--------------------------------------------------------------------------
    // phase 4: construct Ck_Delta and its local cumulative sum
    //--------------------------------------------------------------------------

    // Ck_Delta [pC] = 1 if the pC-th entry is the first in its vector of C, or
    // 0 otherwise.  Then each threadblock computes the inclusive cumulative
    // sum of its chunk of Ck_Delta, overwriting Ck_Delta with its cumulative
    // sum.  Note the spaces (for illustration above) are removed here.

    // input:
    // Ck:  -1 [ 0 0 0 1|4 5 5 5|6 6 6 ]
    //           ^     ^ ^ ^     ^---------start of vectors in C
    // output:
    // Ck_Delta as 0/1:
    //       0 [ 1 0 0 1|1 1 0 0|1 0 0 ]
    // Ck_Delta as inclusive cumsum, per chunk of C:
    //       0 [ 1 1 1 2|1 2 2 2|1 1 1 ]
    // ChunkSum of C:
    //       0 [       2|      2|    1 ]

    // # of chunks in C:
    int64_t nchunks_in_C = (cnz + CHUNK_SIZE - 1) >> LOG2_CHUNK_SIZE ;

    // using W_0 as workspace for Ck_Delta
    GB_Cj_TYPE *Ck_Delta = ((GB_Cj_TYPE *) W_0) + 1 ;
    Ck_Delta [-1] = 0 ;

    // KERNEL LAUNCH 3: phase4
    GB_cuda_select_sparse_phase4 <<<grid, block, 0, stream>>>
        (/* outputs: */ Ck_Delta, ChunkSum,
         /* inputs: */  Ck, cnz, nchunks_in_C) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ck (in W_3) no longer needed

    //--------------------------------------------------------------------------
    // phase 5: construct global cumsum of Ck_Delta on the CPU
    //--------------------------------------------------------------------------

    // overwrite ChunkSum [0..nchunks_in_C] with its cumulative sum
    int64_t cnvec = 0 ;
    for (int64_t chunk = 0 ; chunk < nchunks_in_C ; chunk++)
    {
        // get the # of entries found by this threadblock
        int64_t s = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = cnvec ;
        cnvec += s ;
    }
    ChunkSum [nchunks_in_C] = cnvec ;

    // ChunkSum of C, before the cumsum:
    //       0 [       2|      2|    1 ]
    // ChunkSum of C, after the cumsum, where ChunkSum [nchunks_in_C] = 5
    // are the final number of nonempty vectors of C:
    //       0 [       0|      2|    4 ]  5

    //--------------------------------------------------------------------------
    // phase 6: construct Cp and Ch
    //--------------------------------------------------------------------------

    // The caller has already allocated C->p, C->h for
    // a user-returnable empty hypersparse matrix.
    // Free them here before updating.
    GB_FREE_MEMORY (&(C->p), C->p_size) ;
    GB_FREE_MEMORY (&(C->h), C->h_size) ;

    // Allocate Cp and Ch
    C->plen = cnvec ;
    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;
    C->p = (GB_Cp_TYPE *) GB_MALLOC_MEMORY (C->plen+1, sizeof (GB_Cp_TYPE),
        &(C->p_size)) ;
    C->h = (GB_Cj_TYPE *) GB_MALLOC_MEMORY (C->plen, sizeof (GB_Cj_TYPE),
        &(C->h_size)) ;
    if (C->p == NULL || C->h == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // input:
    // Ck_Delta as inclusive cumsum, per chunk of C:
    //       0 [ 1 1 1 2|1 2 2 2|1 1 1 ]
    // ChunkSum of C:
    //       0 [       2|      2|    1 ]

    // KERNEL LAUNCH 5: phase6
    GB_cuda_select_sparse_phase6 <<<grid, block, 0, stream>>>
        (/* outputs are C->p and C->h: */ C,
         /* inputs: */ Ck_Delta, ChunkSum,
         #if ( GB_A_IS_HYPER )
         A,
         #endif
         cnz, nchunks_in_C) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

