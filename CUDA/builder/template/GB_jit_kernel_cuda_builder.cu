//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_MEMORY (&W_0, W_0_size) ;       \
    GB_FREE_MEMORY (&W_1, W_1_size) ;       \
    GB_FREE_MEMORY (&W_2, W_2_size) ;       \
    GB_FREE_MEMORY (&W_3, W_3_size) ;       \
    GB_FREE_MEMORY (&W_4, W_4_size) ;       \
    GB_FREE_MEMORY (&W_5, W_5_size) ;       \
    GB_FREE_MEMORY (&W_6, W_6_size) ;       \
    GB_FREE_MEMORY (&W_7, W_7_size) ;       \
    GB_FREE_MEMORY (&W_8, W_8_size) ;       \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_Matrix_free (&T) ;                   \
    GB_FREE_WORKSPACE ;                     \
}

using namespace cooperative_groups ;
#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_threadblock_sum_uint64.cuh"

#include <cuda/std/tuple>

#if 0
#define ABORT(msg) {  \
    printf ("Abort! line %d, %s\n", __LINE__, msg) ; \
    fflush (stdout) ; abort ( ) ; }
#endif

//------------------------------------------------------------------------------
// typedefs
//------------------------------------------------------------------------------

// GB_key_t: sorting key type for CUB radix sort

#if GB_BUILD_MATRIX

    #if (GB_KEY_BITS == 32)

#if 1
        // GB_key_t is a single uint64_t, set to ((j << 32) | i);
        // GB_KEY_TYPE is uint32_t
        typedef uint64_t GB_key_t ;
        #define GB_KEY_LOAD(Key_in,p,i1,j1)                             \
            Key_in [p] = ((((uint64_t) (j1)) << 32) | ((uint64_t) i1))
        #define GB_KEY_UNLOAD_I(Key_out,p,i1)                           \
            uint32_t i1 = (uint32_t) (Key_out [p] & 0xFFFFFFFF) ;
        #define GB_KEY_UNLOAD_J(Key_out,p,j1)                           \
            uint32_t j1 = (uint32_t) (Key_out [p] >> 32) ;

#else

        // FIXME: this decomposer object doesn't work:

        // GB_key_t requires a struct
        // GB_KEY_TYPE is uint32_t
        struct GB_key_t
        {
            uint32_t j ;
            uint32_t i ;
            GB_key_t ( ) = default ;
        } ;

        #define GB_KEY_LOAD(Key_in,p,i1,j1)             \
            Key_in [p].i = (uint32_t) i1 ;              \
            Key_in [p].j = (uint32_t) j1 ;
        #define GB_KEY_UNLOAD_I(Key_out,p,i1)           \
            uint32_t i1 = (uint32_t) Key_out [p].i ;
        #define GB_KEY_UNLOAD_J(Key_out,p,j1)           \
            uint32_t j1 = (uint32_t) Key_out [p].j ;

        struct GB_key_decomposer
        {
            __host__ __device__
            cuda::std:tuple <uint32_t&, uint32_t&> operator()(GB_key_t &key)
            const
            {
                return {key.j, key.i} ;
            }
        } ;

#endif

    #else

        // GB_key_t requires a struct
        // GB_KEY_TYPE is uint64_t
        typedef struct
        {
            uint64_t i ;
            uint64_t j ;
        }
        GB_key_t ;
        #define GB_KEY_LOAD(Key_in,p,i1,j1)             \
            Key_in [p].i = (uint64_t) i1 ;              \
            Key_in [p].j = (uint64_t) j1 ;
        #define GB_KEY_UNLOAD_I(Key_out,p,i1)           \
            uint64_t i1 = (uint64_t) Key_out [p].i ;
        #define GB_KEY_UNLOAD_J(Key_out,p,j1)           \
            uint64_t j1 = (uint64_t) Key_out [p].j ;

        // FIXME: need a decomposer object here

        #error "not yet working"

    #endif

#else

    // Buidling a GrB_Vector, so j is not used. 
    // GB_key_t is not a struct; just a plain uint32_t or uint64_t
    typedef GB_KEY_TYPE GB_key_t ;
    #define GB_KEY_LOAD(Key_in,p,i1,j1)                 \
        Key_in [p] = (GB_KEY_TYPE) i1 ;
    #define GB_KEY_UNLOAD_I(Key_out,p,i1)               \
        GB_KEY_TYPE i1 = Key_out [p] ;
    #define GB_KEY_UNLOAD_J(Key_out,p,j1) ;

#endif

#define GB_KEY_UNLOAD(Key_out,p,i1,j1)                  \
    GB_KEY_UNLOAD_I (Key_out, p, i1) ;                  \
    GB_KEY_UNLOAD_J (Key_out, p, j1) ;

//------------------------------------------------------------------------------
// geometry
//------------------------------------------------------------------------------

#define CHUNKSIZE           GB_CUDA_BUILDER_CHUNKSIZE
#define LOG2_CHUNKSIZE      GB_CUDA_BUILDER_CHUNKSIZE_LOG2
#define BLOCKDIM            GB_CUDA_BUILDER_BLOCKDIM
#define ITEMS_PER_THREAD    (CHUNKSIZE / BLOCKDIM)

// Int can be uint16_t if CHUNKSIZE is < 65,535
#define Int uint16_t

//------------------------------------------------------------------------------
// GB_cuda_builder_phase1:
//------------------------------------------------------------------------------

// phase1 loads the (I,J) tuples into the Key_in workspace, and checks the
// indices (I,J) to ensure they are in range.  It sets the global (*bad) scalar
// to the # of tuples that are out of range.  Note that the CPU builder also
// returns the first invalid indices for the error message returned to the user
// application; this kernel does not do that.

// TODO: phase1 could also check if the (I,J) indices are already in sorted
// order, which would allow the method to skip the CUB radix sort in phase2.

__global__ void GB_cuda_builder_phase1
(
    // output
    GB_key_t *Key_in,   // size nvals+1: Key_in [-1...nvals-1]
//  uint64_t *ok,       // if true: (I,J) are valid; false: (I,J) out of range
    uint64_t *bad,      // count of # of invalid tuples (zero on input)
    // input
    const GB_I_TYPE *__restrict__ I, // size nvals
    #if GB_BUILD_MATRIX
    const GB_J_TYPE *__restrict__ J, // size nvals, NULL if C is a vector
    #endif
    int64_t vlen,       // vector-length dimension of C (for I indices)
    int64_t vdim,       // vector-dim dimension of C (for J indices)
    int64_t nvals       // # of tuples in (I,J,X)
)
{

    //--------------------------------------------------------------------------
    // load the (I,J) tuples into the Key_in workspace
    //--------------------------------------------------------------------------

    #if 0
    bool my_ok = true ;
    #endif
    uint64_t my_bad = 0 ;

    for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                 p < nvals ;
                 p += blockDim.x * gridDim.x)       // grid-block-stride loop
    {
        // get the indices
        GB_I_TYPE i = I [p] ;
        #if GB_BUILD_MATRIX
        GB_J_TYPE j = J [p] ;
        #endif

        // count the # of tuples out of range
        #if GB_BUILD_MATRIX
        my_bad += (uint64_t) ((i >= vlen) + (j >= vdim)) ;
        #else
        my_bad += (uint64_t) ((i >= vlen)) ;
        #endif

        // load the indices into Key_in [p]
        GB_KEY_LOAD (Key_in, p, i, j) ;
    }

    //--------------------------------------------------------------------------
    // compute the global count of tuples out of range
    //--------------------------------------------------------------------------

    my_bad = GB_cuda_threadblock_sum_uint64 (my_bad) ;
    if (threadIdx.x == 0)
    {
        GB_cuda_atomic_add <uint64_t> (bad, my_bad) ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_builder_phase3
//------------------------------------------------------------------------------

// phase3 looks for duplicates in the sorted Key_out array.  It constructs a
// Map array which tells each entry in (Key_out,Sx) where it appears in C,
// after a cumulative sum.  This part of the phase is nearly identical to
// CUDA/select phase1.  builder/phase3 must also find the leading entry in each
// vector of the output matrix C.  It does this with JDelta, which is a
// cumulative sum of the number of leading entries in each chunk.

// Compare with select/phase1

__global__ void GB_cuda_builder_phase3
(
    // outputs
    Int *Map,               // size nvals+1, in Map [-1...nvals-1]
    GB_Tp_TYPE *ChunkSum,   // size nchunks+1,
                            // in ChunkSum [-1..nchunks]
    #if GB_BUILD_MATRIX
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    // inputs, not modified, except for the Key_out [-1] sentinel value:
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ Int Local_Map [CHUNKSIZE] ;
    #if GB_BUILD_MATRIX
    __shared__ Int Local_JDelta [CHUNKSIZE] ;
    #endif

    // cub::Block* workspace:
    GB_CUB_BLOCK_WORKSPACE (W, Int, BLOCKDIM, ITEMS_PER_THREAD) ;

    //--------------------------------------------------------------------------
    // the first thread of the threadblock fills in the sentinal values
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        memset (&(Key_out [-1]), 0xFF, sizeof (GB_key_t)) ;
        Map [-1] = 0 ;
        ChunkSum [-1] = 0 ;
        #if GB_BUILD_MATRIX
        JDelta [-1] = 0 ;
        JDeltaSum [-1] = 0 ;
        #endif
    }

    // this_thread_block ( ).sync ( ) ; not needed since the thread that wrote
    // the Key_out [-1] entry is the only thread that reads it.

    //--------------------------------------------------------------------------
    // compute each local chunk of Map
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks ;
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE ;
        int64_t my_chunk_size ;
        // this computation is just the 2nd #if case of select/phase1:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

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
            // determine if the p-th entry is 1st of duplicates, or leading
            //------------------------------------------------------------------

            // get the indices
            GB_KEY_UNLOAD (Key_out, p-1, iprev, jprev) ;
            GB_KEY_UNLOAD (Key_out, p,   i,     j    ) ;
            #if GB_BUILD_MATRIX
            bool leading = (j != jprev) ;
            #endif
            // keep = 1 if (i,j) is unique, 0 if duplicate
            bool keep = (i != iprev)
                #if GB_BUILD_MATRIX
                || leading
                #endif
                ;
            Local_Map [pdelta] = keep ;         // 1 if 1st in seq of dupls
            #if GB_BUILD_MATRIX
            Local_JDelta [pdelta] = leading ;   // 1 if leading entry of vector
            #endif
            #if 0
            printf ("thread %d (i,j) = (%u, %u), (iprev,jprev) = (%u, %u)"
                " keep: %d, leading: %d\n",
                threadIdx.x, i, j, iprev, jprev, keep, leading) ;
            #endif
        }

        //----------------------------------------------------------------------
        // the remainder is similar to select/phase1:
        //----------------------------------------------------------------------

        // clear the unused part of the Local_Map and Local_JDelta
        for ( ; pdelta < CHUNKSIZE ;
                pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
            #if GB_BUILD_MATRIX
            Local_JDelta [pdelta] = 0 ;
            #endif
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map and Local_JDelta
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]) is computed.

        // Similarly, JDelta [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum
        // of Local_JDelta, where Local_JDelta [i] = sum (Local_JDelta [0:i])
        // is computed.

        this_thread_block ( ).sync ( ) ;
        Int t_block_aggregate ;
        #if GB_BUILD_MATRIX
        Int s_block_aggregate ;
        #endif

#if 0
        // This entire phase computes the following:
        if (threadIdx.x == blockDim.x - 1)
        {
            // construct Map and ChunkSum
            for (int i = 1 ; i < CHUNKSIZE ; i++)
            {
                Local_Map [i] += Local_Map [i-1] ;
            }
            for (int i = 0 ; i < CHUNKSIZE ; i++)
            {
                Map [pfirst + i] = Local_Map [i] ;
            }
            t_block_aggregate = Local_Map [CHUNKSIZE-1] ;
            ChunkSum [chunk] = t_block_aggregate ;

            // construct JDelta and JDeltaSum
            #if GB_BUILD_MATRIX
            for (int i = 1 ; i < CHUNKSIZE ; i++)
            {
                Local_JDelta [i] += Local_JDelta [i-1] ;
            }
            for (int i = 0 ; i < CHUNKSIZE ; i++)
            {
                JDelta [pfirst + i] = Local_JDelta [i] ;
            }
            s_block_aggregate = Local_JDelta [CHUNKSIZE-1] ;
            printf ("chunk: %d, s_block_agg %d\n",
                (int) chunk, (int) s_block_aggregate) ;
            JDeltaSum [chunk] = s_block_aggregate ;
            #endif
        }

#else
        Int t [ITEMS_PER_THREAD] ;

        BlockLoad (W.load).Load (Local_Map, t) ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W.scan).InclusiveSum (t, t, t_block_aggregate) ;
        this_thread_block ( ).sync ( ) ;
        BlockStore (W.store).Store (Map + pfirst, t) ;
        this_thread_block ( ).sync ( ) ;

        #if GB_BUILD_MATRIX
        BlockLoad (W.load).Load (Local_JDelta, t) ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W.scan).InclusiveSum (t, t, s_block_aggregate) ;
        this_thread_block ( ).sync ( ) ;
        BlockStore (W.store).Store (JDelta + pfirst, t) ;
        this_thread_block ( ).sync ( ) ;
        #endif

#endif

        // finally, the aggregate sums are written to ChunkSum and JDeltaSum
        if (threadIdx.x == blockDim.x - 1)
        {
            #if 0
            printf ("phase3, set chunk: %ld: %d, %d\n", chunk,
                t_block_aggregate, s_block_aggregate);
            #endif
            ChunkSum  [chunk] = t_block_aggregate ;
            #if GB_BUILD_MATRIX
            JDeltaSum [chunk] = s_block_aggregate ;
            #endif
        }


        this_thread_block ( ).sync ( ) ;

    }
}

//------------------------------------------------------------------------------
// GB_cuda_builder_phase5
//------------------------------------------------------------------------------

// phase5 constructs the output matrix T (Tp, Th, Ti, and Tx) from the
// (Key_out,Sx) tuples, summing up duplicates (if any).

// compare with select/phase3 and select/phase6

__global__ void GB_cuda_builder_phase5
(
    // outputs
    GrB_Matrix T,
    // inputs, not modified:
    Int *Map,               // size nvals+1, in Map [-1...nvals-1]
    GB_Tp_TYPE *ChunkSum,   // size nchunks+1,
                            // in ChunkSum [-1..nchunks]
    #if GB_BUILD_MATRIX
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    GB_Sx_TYPE *Sx,         // size nvals+1: Sx  [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // get T->p, T->h, T->i, and T->x, shifting down by 1 since Map is 1-based
    //--------------------------------------------------------------------------

    GB_Tp_TYPE *__restrict__ Tp = (GB_Tp_TYPE *) T->p ; Tp-- ;
    #if GB_BUILD_MATRIX
    GB_Tj_TYPE *__restrict__ Th = (GB_Tj_TYPE *) T->h ; Th-- ;
    #endif
    GB_Ti_TYPE *__restrict__ Ti = (GB_Ti_TYPE *) T->i ; Ti-- ;
    #if !GB_ISO_BUILD
    GB_Tx_TYPE *__restrict__ Tx = (GB_Tx_TYPE *) T->x ; Tx-- ;
    #endif

    // TODO: if no duplicates have been found, Map and ChunkSum are not
    // needed.  Just copy all of Key_out [0..nvals-1].i to Ti [0..nvals-1]

    #if 0
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf ("Chunks at start of phase5:\n") ;
        for (int64_t chunk = -1 ; chunk < nchunks ; chunk++)
        {
            printf ("ChunkSum [%ld] = %ld,  JDeltaSum [%ld] = %ld\n",
                chunk, (int64_t) ChunkSum [chunk],
                chunk, (int64_t) JDeltaSum [chunk]) ;
        }
    }
    this_thread_block ( ).sync ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // copy the entries from (Key_out,Sx) into Tp, Th, Ti, and Tx, summing dupls
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks ;
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE ;
        int64_t my_chunk_size ;
        // this computation is just the 2nd #if case of select/phase3:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;
//      printf ("chunk: %ld, pfirst: %ld, plast %ld\n",  chunk, pfirst, plast) ;

        //----------------------------------------------------------------------
        // copy the entries, sum duplicates, and construct Tp and Th
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)       // block-stride loop
        {

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // copy the entries, summing the duplicates
            //------------------------------------------------------------------

            // get the position pT in C of the p-th tuple in (Key_out,Sx)
            GB_Tp_TYPE pT = Map [p  ] + ChunkSum [chunk] ;
            // get the position p0 in C of the (p-1)-st tuple in (Key_out,Sx)
            GB_Tp_TYPE p0 = Map [p-1] + ChunkSum [chunk - (pdelta == 0)] ;
            #if 0
            printf ("pfirst %ld, pdelta %ld, p: %ld, p0: %d, pT: %d\n",
                pfirst, pdelta, p, p0, pT) ;
            #endif
            if (p0 < pT)
            {
                // This entry is the first in a sequence of duplicates (perhaps
                // just a single entry with no duplicates)
                // Ti [pT] = Key_out [p].i ;
                GB_KEY_UNLOAD_I (Key_out, p, i1) ;
                Ti [pT] = (GB_Ti_TYPE) i1 ;

                #if !GB_ISO_BUILD
                GB_BLD_COPY (Tx, pT, Sx, p) ; // Tx [pT] = Sx [p]

                #if !defined (GB_DUP_IS_FIRST)
                // Sum up all duplicate entries, in order.  This can cross over
                // into subsequent chunks of (Key_out,Sx).  Warp divergence is
                // expected, but it should be OK since only a modest O(1)
                // number of duplicate are expected for each unique T(i,j)
                // entry.
                int64_t chunk2 = chunk ;
                bool is_duplicate = true ;
                for (uint64_t p2 = p+1 ; (p2 < nvals) && is_duplicate ; p2++)
                {
                    // get the next entry: increment the chunk2 of p2
                    chunk2 += ((p2 & (CHUNKSIZE-1)) == 0) ;
                    GB_Tp_TYPE pdupl = Map [p2] + ChunkSum [chunk2] ;
                    is_duplicate = (pT == pdupl) ;
                    if (is_duplicate)
                    {
                        // Tx [pT] += Sx [pdupl]
                        #if 0
                        printf ("p: %ld, p2: %ld, nvals %ld, pT: %ld, "
                            "pdupl: %ld, got duplicate\n",
                            p, p2, nvals, (uint64_t) pT, (uint64_t) pdupl) ;
                        #endif
                        GB_BLD_DUP (Tx, pT, Sx, pdupl) ;
                    }
                }
                #endif
                #endif
            }

            //------------------------------------------------------------------
            // construct Tp and Th, if T is a matrix (skip if T is a vector)
            //------------------------------------------------------------------

            #if GB_BUILD_MATRIX
            GB_Tp_TYPE kT = JDelta [p  ] + JDeltaSum [chunk] ;
            GB_Tp_TYPE k0 = JDelta [p-1] + JDeltaSum [chunk - (pdelta == 0)] ;
            #if 0
            printf ("p %ld, k0: %ld, kT: %ld, k0 < kT: %d\n",
                p, (int64_t) k0, (int64_t) kT, k0 < kT) ;
            #endif
            if (k0 < kT)
            {
                // The p-th entry is the leading entry of the kT-th vector of T
                #if 0
                printf ("got lead: p: %ld:  k0 %d kT %d, pT-1: %d\n",
                    p, k0, kT, pT-1) ;
                #endif
                Tp [kT] = pT - 1 ;      // shift by 1 since pT is 1-based
                // Th [kT] = Key_out [p].j ;
                GB_KEY_UNLOAD_J (Key_out, p, j1) ;
                Th [kT] = j1 ;
            }
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // T->nvec is 0-based, so increment Tp to undo the Tp-- done above
        Tp++ ;
        #if GB_BUILD_MATRIX
        Tp [T->nvec] = T->nvals ;
        #else
        Tp [0] = 0 ;
        Tp [1] = T->nvals ;
        #endif
    }
}

//------------------------------------------------------------------------------
// builder, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_new_bix) ;
    GB_GET_CALLBACK (GB_Matrix_free) ;
    #endif

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // GB_I_TYPE and GB_J_TYPE are uint32_t or uint64_t
    GB_I_TYPE  *__restrict__ I = (GB_I_TYPE  *) I_input ;
    GB_J_TYPE  *__restrict__ J = (GB_J_TYPE  *) J_input ;
    GB_Sx_TYPE *__restrict__ X = (GB_Sx_TYPE *) X_input ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    (*Thandle) = NULL ;
    GrB_Matrix T = NULL ;
    GrB_Info info = GrB_SUCCESS ;

    // workspace needed for CUB radix sort of (Key_in,X):
    void *W_0 = NULL ; size_t W_0_size = 0 ;    // size nvals+1: Key_in
    void *W_1 = NULL ; size_t W_1_size = 0 ;    // size nvals+1: Key_out
    void *W_2 = NULL ; size_t W_2_size = 0 ;    // size nvals+1: Sx (or NULL)
    void *W_3 = NULL ; size_t W_3_size = 0 ;    // size nvals+1: CUB workspace

    // when the CUB radix sort is done, Key_in and the CUB workspace can
    // be freed.

    // workspace needed after CUB radix sort:
    void *W_4 = NULL ; size_t W_4_size = 0 ;    // size nvals+1: Map
    void *W_5 = NULL ; size_t W_5_size = 0 ;    // size nchunks+2: ChunkSum
    void *W_6 = NULL ; size_t W_6_size = 0 ;    // size nvals+1: JDelta
    void *W_7 = NULL ; size_t W_7_size = 0 ;    // size nchunks+2: JDeltaSum

    void *W_8 = NULL ; size_t W_8_size = 0 ;    // size 2: scalar workspace

    // # of entries, chunks, and vectors of T
    int64_t tnz = 0 ;   // # of unique tuples, and # of entries in T
    int64_t tnvec = 0 ; // # of vectors of T
    int64_t nchunks = (nvals + CHUNKSIZE - 1) >> LOG2_CHUNKSIZE ;
    // printf ("nchunks: %ld\n", nchunks) ;

    dim3 grid (gridsz) ;        // = min (ceil (nvals/CHUNKSIZE), 256*(#sms))
    dim3 block1 (BLOCKDIM) ;

    cudaStream_t mystream, mystream3 ;
    CUDA_OK (cudaStreamCreate (&mystream)) ;
    CUDA_OK (cudaStreamCreate (&mystream3)) ;

    //--------------------------------------------------------------------------
    // phase1: load the Key_in workspace and check if indices are in range
    //--------------------------------------------------------------------------

    // Example, assuming a chunksize of 4, with 14 tuples and 3 duplicates
    // marked with (*).  The first entry of each set of duplicates is marked
    // with (^), but this is not detected in phase1.

    // J:          [ 0 1 0 0 | 0 1 1 1 | 2 2 4 1 | 4 4 ]
    // I:          [ 0 3 1 2 | 1 4 5 5 | 3 4 3 5 | 0 1 ]
    // dupl:             ^     *   ^ *         *

    // builder/phase1 loads the (I,J) tuples into Key_in and ensures the
    // indices are in range.

    // TODO: the check for valid indices could be skipped if this method knows
    // its I,J inputs are already valid.

    // allocate Key_in and W_8 workspace
    W_0 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_key_t), &W_0_size) ;
    W_8 = GB_MALLOC_MEMORY (2, sizeof (uint64_t), &W_8_size) ;
    if (W_0 == NULL || W_8 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one so Key_in [-1...nvals-1] can be used
    GB_key_t *Key_in = ((GB_key_t *) W_0) + 1 ;

    #if 0
    printf ("W_0: %p Key_in %p difference: %lu\n",
        W_0, Key_in, (uint64_t) (Key_in - ((GB_key_t *) W_0))) ;
    printf ("\n\n============================== builder phase 1\n\n") ;
    #endif

    #if 0
    #if GB_BUILD_MATRIX
    for (int64_t p = 0 ; p < nvals ; p++)
    {
        int64_t i = I [p] ;
        int64_t j = J [p] ;
        printf ("tuple [%ld] = (%ld, %ld)\n", p, i, j) ;
    }
    #endif
    #endif

    // get 2 uint64_t scalars from the W_8 workspace
    uint64_t *bad = ((uint64_t *) W_8) ;
//  uint64_t *ok  = ((uint64_t *) W_8) + 1 ;
    (*bad) = 0 ;
//  (*ok) = 1 ;

    GB_cuda_builder_phase1 <<<grid, block1, 0, mystream>>>
        (/* outputs: */ Key_in, /* ok, */ bad,
         /* inputs: */ I,
            #if GB_BUILD_MATRIX
            J,
            #endif
            vlen, vdim, nvals) ;

    cudaError_t err1 = cudaGetLastError ( ) ;
//  printf ("phase1 cuda error %d\n", err1) ;
    CUDA_OK (err1) ;
    CUDA_OK (cudaStreamSynchronize (mystream)) ;
//  (*ok) = (*bad == 0) ;
//  printf ("phase1 sync, ok: %lu, bad: %lu\n", *ok, *bad) ;

    // HACK: repeat phase1 on the CPU
    #if 0
    bool my_ok = true ;
    uint64_t my_bad = 0 ;
    for (int64_t p = 0 ; p < nvals ; p ++)
    {
        // get the indices
        GB_I_TYPE i = I [p] ;
        #if GB_BUILD_MATRIX
        GB_J_TYPE j = J [p] ;
        #endif
        // check if the indices are in range
        my_ok = my_ok
            #if GB_BUILD_MATRIX
            && (j < vdim)
            #endif 
            && (i < vlen) ;
        my_bad += (!my_ok) ;
        // load the indices into Key_in [p]
        uint64_t key = (uint64_t) i ;
        #if GB_BUILD_MATRIX
        key = (((uint64_t) (j)) << 32) | key ;
        #endif
        if (Key_in [p] != key) ABORT ("phase 1 failed\n") ;
    }
    (*bad) = my_bad ;
    (*ok) = (*bad == 0) ;
    printf ("phase1 CPU, ok: %lu, bad: %lu\n", *ok, *bad) ;
    #endif

    // after the CUDA kernel launch is done, check if the (I,J) indices are OK
    GB_OK (((*bad) == 0) ? GrB_SUCCESS : GrB_INVALID_INDEX) ;

    #if 0
    #if GB_BUILD_MATRIX
    for (int64_t p = 0 ; p < nvals ; p++)
    {
        GB_KEY_UNLOAD (Key_in, p, i, j) ;
        printf ("Key_in [%ld] = (%ld, %ld)\n", p, (int64_t) i, (int64_t) j) ;
    }
    #endif
    #endif

    // TODO: The original (I,J) inputs are no longer needed at this point.  If
    // (I,J) can be modified and freed on return (see the CPU GB_builder
    // method), then they can be used as workspace or as Ti, Tx components of
    // the output matrix.

    //--------------------------------------------------------------------------
    // phase2: CUB radix sort of (Key_in,X) to obtain (Key_out,Sx)
    //--------------------------------------------------------------------------

    // builder/phase2 sorts the tuples in (Key_in,X) to obtain (Key_out,Sx).
    // The dupl state (of first entry in sequence of duplicates), or leading
    // state (if the entry is the first in its vector of C) are not yet
    // computed, but shown below for reference.  The (@) denotes the leading
    // entry of each vector in C.

    // output of builder/phase2 (Sx values not shown):
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4 ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C

    // TODO: phase2 could be skipped if phase1 detects (I,J) are already sorted
    // in phase1, or if it knows (I,J) are already sorted on input.  In this
    // case, Key_out = Key_in can be done instead of allocating Key_out.

    #if 0
    printf ("\n\n============================== builder phase 2\n\n") ;
    printf ("nvals %ld, iso: %d\n", nvals, GB_ISO_BUILD) ;
    #endif

    // allocate Key_out, Sx, and CUB temporary workspace
    W_1 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_key_t), &W_1_size) ;
    // printf ("W_1_size: %lu\n", (uint64_t) W_1_size) ;
    #if !GB_ISO_BUILD
    W_2 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_Sx_TYPE), &W_2_size) ;
    #endif
    if (W_1 == NULL || (!GB_ISO_BUILD && W_2 == NULL))
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one so Key_out [-1...nvals-1], etc can be used
    GB_key_t *Key_out = ((GB_key_t   *) W_1) + 1 ;
    GB_Sx_TYPE *Sx    = ((GB_Sx_TYPE *) W_2) + 1 ;

    // determine the amount of workspace needed by CUB radix sort
    #if GB_ISO_BUILD
    CUDA_OK (cub::DeviceRadixSort::SortKeys (
        /* temp storage: */ W_3, W_3_size,
        Key_in, Key_out, nvals,
        #if 0
        GB_key_decomposer { },
        #endif
        /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
        mystream)) ;
    #else
    CUDA_OK (cub::DeviceRadixSort::SortPairs (
        /* temp storage: */ W_3, W_3_size,
        Key_in, Key_out, /* values in: */ X, /* values out: */ Sx, nvals,
        #if 0
        GB_key_decomposer { },
        #endif
        /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
        mystream)) ;
    #endif

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (mystream)) ;

    // allocate workspace for CUB radix sort
    W_3 = GB_MALLOC_MEMORY (W_3_size+1, 1, &W_3_size) ;
    // printf ("W_3_size for radix sort: %lu\n", (uint64_t) W_3_size) ;
    if (W_3 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // sort (Key_in,X) to get (Key_out,Sx)
    #if GB_ISO_BUILD
    CUDA_OK (cub::DeviceRadixSort::SortKeys (
        /* temp storage: */ W_3, W_3_size,
        Key_in, Key_out, nvals,
        #if 0
        GB_key_decomposer { },
        #endif
        /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
        mystream)) ;
    #else
    CUDA_OK (cub::DeviceRadixSort::SortPairs (
        /* temp storage: */ W_3, W_3_size,
        Key_in, Key_out, /* values in: */ X, /* values out: */ Sx, nvals,
        #if 0
        GB_key_decomposer { },
        #endif
        /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
        mystream)) ;
    #endif

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (mystream)) ;

    // Key_in and CUB workspace no longer needed
    #if 0
    // FIXME: add this back in
    GB_FREE_MEMORY (&W_0, W_0_size) ;
    GB_FREE_MEMORY (&W_3, W_3_size) ;
    Key_in = NULL ;
    #endif

    // sorted tuples are now in (Key_out,Sx) 

    #if 0
    #if GB_BUILD_MATRIX
    for (int64_t p = 0 ; p < nvals ; p++)
    {
        GB_KEY_UNLOAD (Key_out, p, i, j) ;
        printf ("Key_out [%ld] = (%ld, %ld)\n", p, (int64_t) i, (int64_t) j) ;
    }
    #endif
    #endif

    //--------------------------------------------------------------------------
    // phase3: look for duplicates (compare with phase1 of CUDA/select)
    //--------------------------------------------------------------------------

    #if 0
    printf ("\n\n============================== builder phase 3\n\n") ;
    #endif

    // builder/phase3 determines which entries are the first in a sequence of
    // duplicates (output: Map), and which entries are the first in their
    // respective vectors (output: JDelta).  The output after builder/phase3 is
    // shown below, where LMap = Local_Map (a temporary shared array in each
    // threadblock of phase3).  Map is the cumsum of each chunk of LMap.  Note
    // the padding of LMap, Map, LJDelta, and JDelta.  Any given sequence of
    // duplicates can span across the chunks (the (1,5) entry does so).

    // LJDelta is 1 if the entry is first in its vector (a leading entry), held
    // in a temporary shared array in each threadblock.  It is computed from
    // Key_out.j only and is not affected by the presence of duplicates.
    // JDelta is the cumsum of chunk of LJDelta. 

    // inputs:
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4     ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3     ]
    // output:
    // LMap:         [ 1 1 0 1 | 1 1 1 0 | 0 1 1 1 | 1 1 - - ]
    // Map:        0 [ 1 2 2 3 | 1 2 3 3 | 0 1 2 3 | 1 2 2 2 ]
    // LJDelta:    0 [ 1 0 0 0 | 1 0 0 0 | 0 1 0 1 | 0 0 0 0 ]
    // JDelta:     0 [ 1 1 1 1 | 1 1 1 1 | 0 1 1 2 | 0 0 0 0 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C

    // ChunkSum [-1..nchunks] is the # of non-duplicates to be kept in each
    // chunk, with ChunkSum [-1] = 0 and ChunkSum [nchunks] = 0 for now.  This
    // matrix will have 11 total entries, which is the total sum of ChunkSum:
    //            0 [       3 |       3 |       3 |       2 ] 0

    // JDeltaSum [-1..nchunks] is the # of leading entries in each chunk,
    // with JDeltaSum [-1] = 0.  This matrix has 4 unique values of J (0, 1, 2,
    // and 4), which is the total sum of JDeltaSum:
    //            0 [       1 |       1 |       2 |       0 ] 0

    // allocate Map and ChunkSum: for cumsum of 1st entries in sequence of dupls
    W_4 = GB_MALLOC_MEMORY (nvals+1 + CHUNKSIZE, sizeof (Int), &W_4_size) ;
    W_5 = GB_MALLOC_MEMORY (nchunks+2, sizeof (GB_Tp_TYPE), &W_5_size) ;

    // allocate JDelta, JDeltaSum: for cumsum of leading entries of vectors of C
    #if GB_BUILD_MATRIX
    W_6 = GB_MALLOC_MEMORY (nvals+1 + CHUNKSIZE, sizeof (Int), &W_6_size) ;
    W_7 = GB_MALLOC_MEMORY (nchunks+2, sizeof (GB_Tp_TYPE), &W_7_size) ;
    #endif

    // shift by one so Map [-1...nvals-1], etc can be used
    Int *Map              = ((Int        *) W_4) + 1 ;
    GB_Tp_TYPE *ChunkSum  = ((GB_Tp_TYPE *) W_5) + 1 ;
    #if GB_BUILD_MATRIX
    Int *JDelta           = ((Int        *) W_6) + 1 ;
    GB_Tp_TYPE *JDeltaSum = ((GB_Tp_TYPE *) W_7) + 1 ;
    #endif

    #if 0
    #if GB_BUILD_MATRIX
    printf ("\nphase3 input:\n") ;
    for (int64_t p = 0 ; p < nvals ; p++)
    {
        GB_KEY_UNLOAD (Key_out, p, i, j) ;
        printf ("Key_out [%ld] = (%ld, %ld), chunk: %ld\n",
            p, (int64_t) i, (int64_t) j, p / CHUNKSIZE) ;
    }
    #endif
    #endif

    #if 0
    printf ("do phase 3: gridsz %ld, blockdim %ld\n",
        (int64_t) gridsz, (int64_t) BLOCKDIM) ;
    #endif

    GB_cuda_builder_phase3 <<<grid, block1, 0, mystream3>>>
        ( /* outputs: */ Map, ChunkSum,
            #if GB_BUILD_MATRIX
            JDelta, JDeltaSum,
            #endif
          /* inputs: */ Key_out, nvals, nchunks) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (mystream3)) ;

    #if GB_BUILD_MATRIX
    #if 0
    printf ("\nphase3 output:\n") ;
    for (int64_t p = -1 ; p < nvals ; p++)
    {
        GB_KEY_UNLOAD (Key_out, p, i, j) ;
        printf ("\nKey_out [%ld] = (%ld, %ld), chunk: %ld\n",
            p, (int64_t) i, (int64_t) j, p / CHUNKSIZE) ;
        printf ("    Map [%ld] = %d\n", p, Map [p]) ;
        printf ("    JDelta [%ld] = %d\n", p, JDelta [p]) ;
    }
    #endif

    #if 0
    printf ("Chunks after phase3:\n") ;
    bool ok2 = true ;
    for (int64_t chunk = 0 ; chunk < nchunks ; chunk++)
    {
        printf ("ChunkSum [%ld] = %ld,  JDeltaSum [%ld] = %ld\n",
            chunk, (int64_t) ChunkSum [chunk],
            chunk, (int64_t) JDeltaSum [chunk]) ;
        ok2 = ok2 && (ChunkSum [chunk] <= CHUNKSIZE)
                && (JDeltaSum [chunk] <= CHUNKSIZE) ;
    }
    if (!ok2) ABORT ("chunks are bad!\n") ;
    #endif
    #endif

    //--------------------------------------------------------------------------
    // phase4: sum up the unique entries in each chunk (on the CPU)
    //--------------------------------------------------------------------------

    #if 0
    printf ("\n\n============================== builder phase 4\n\n") ;
    #endif

    // compare with phase2 of CUDA/select

    // At the start of phase4, ChunkSum [-1..nchunks] holds the # of
    // unique entries in C from each chunk of (I,J,X), with tnz = 11 unique
    // entries in C for this example:

    // ChunkSum on input (x is 'dont care'):
    //            x [       3 |       3 |       3 |       2 ] 0
    // ChunkSum on output (an exclusive sum):
    //            0 [       0 |       3 |       6 |       9 ] 11

    // JDeltaSum on input:
    //            x [       1 |       1 |       2 |       0 ] 0
    // JDeltaSum on output (an exclusive sum):
    //            0 [       0 |       1 |       2 |       4 ] 4

    ChunkSum [-1] = 0 ;         // sentinel value
    #if GB_BUILD_MATRIX
    JDeltaSum [-1] = 0 ;        // sentinel value
    #endif

    // overwrite ChunkSum [0..gridsz] with its cumulative sum
    for (int64_t chunk = 0 ; chunk < nchunks ; chunk++)
    {
        // get the # of entries found in this chunk
        int64_t t = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = tnz ;
        tnz += t ;

        // get the # of leading entries found in this chunk
        #if GB_BUILD_MATRIX
        int64_t s = JDeltaSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // JDeltaSum [chunk] = original JDeltaSum [0..chunk-1]
        JDeltaSum [chunk] = tnvec ;
        tnvec += s ;
        #endif
    }
    ChunkSum  [nchunks] = tnz ;
    #if GB_BUILD_MATRIX
    JDeltaSum [nchunks] = tnvec ;
    #else
    tnvec = 1 ;
    #endif

    #if 0
    printf ("did phase4: tnvec is %ld\n", tnvec) ;
    printf ("Chunks after phase4:\n") ;
    for (int64_t chunk = -1 ; chunk <= nchunks ; chunk++)
    {
        printf ("ChunkSum [%ld] = %ld",
            chunk, (int64_t) ChunkSum [chunk]) ;
        #if GB_BUILD_MATRIX
        printf (", JDeltaSum [%ld] = %ld\n",
            chunk, (int64_t) JDeltaSum [chunk]) ;
        #endif
        printf ("\n") ;
    }
    #endif

    //--------------------------------------------------------------------------
    // phase5: allocate T and construct it
    //--------------------------------------------------------------------------

    // compare with phase3 of CUDA/select

    #if 0
    printf ("\n\n============================== builder phase 5\n\n") ;
    #endif

    // builder/phase5 allocates the output T matrix (including T->p, T->h,
    // T->i, and T->x arrays), then moves the data from (Key_out,Sx) into
    // (Tp,Th,Ti,Tx), applying the dup operator to "sum" up the values of the
    // duplicates as it does so.  Each sequence of duplicates is a handled by a
    // single thread in a single threadblock.  This assumes there are not many
    // duplicates for each entry.

    // input:
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4     ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3     ]
    // Map:        0 [ 1 2 2 3 | 1 2 3 3 | 0 1 2 3 | 1 2 2 2 ]
    // JDelta:     0 [ 1 1 1 1 | 1 1 1 1 | 0 1 1 2 | 0 0 0 0 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C
    // ChunkSum:   0 [       0 |       3 |       6 |       9 ] 11
    // JDeltaSum   0 [       0 |       1 |       2 |       4 ] 4

    // Assume for this example that Sx [...] = 1, and dup is "+", so the value
    // of Tx is simply the number of duplicates of that entry.

    // The position where the p-th entry in (I,J,X) appears in T is given by
    // pT = Map [p] + ChunkSum [chunk], where pT is a 1-based index.  If this
    // position differs from the position of the (p-1)st entry in (I,J,X),
    // then the entry is the first unique tuple in its set of duplicates.

    // output with "|" denoting the chunks:
    // Ti:           [ 0 1 2 | 3 4 5 | 3 4 0 | 1 3 ]
    // Tx:           [ 1 2 1 | 1 1 3 | 1 1 1 | 1 1 ]
    // Tp:           [ 0 3 6 8 11 ]
    // Th:           [ 0 1 2 4 ]

    // output repeated but with "|" denoting the vectors of T:
    // Ti:           [ 0 1 2 | 3 4 5 | 3 4 | 0 1 3 ]
    // Tx:           [ 1 2 1 | 1 1 3 | 1 1 | 1 1 1 ]
    // Tp:           [ 0       3       6     8       11 ] of length T->nvec+1
    // Th:           [ 0       1       2     4          ] of length T->nvec

    // allocate the T matrix as hypersparse, with tnz entries and tnvec vectors,
    // or as sparse if T is a typecasted GrB_Vector.
    GB_OK (GB_new_bix (&T, ttype, vlen, vdim,
        /* Ap_option: */ (tnz == 0) ? GB_ph_calloc : GB_ph_malloc,
        is_csc,
        #if GB_BUILD_MATRIX
        /* sparsity: */ GxB_HYPERSPARSE,
        /* bitmap_calloc: */ false,
        /* hyper_switch: */ GB_ALWAYS_HYPER,
        #else
        /* sparsity: */ GxB_SPARSE,
        /* bitmap_calloc: */ false,
        /* hyper_switch: */ GB_NEVER_HYPER,
        #endif
        /* plen: */ tnvec,
        /* nzmax: */ tnz+2,
        /* numeric: */ true,
        /* A_iso: */ GB_ISO_BUILD,
        /* p_is_32: */ (GB_Tp_BITS == 32),
        /* j_is_32: */ (GB_Tj_BITS == 32),
        /* i_is_32: */ (GB_Ti_BITS == 32))) ;

    T->nvals = tnz ;
    T->magic = GB_MAGIC ;
    T->nvec = tnvec ;
    T->nvec_nonempty = tnvec ;

    #if 0
    printf ("did phase5 new_bix: T->nvec = %ld\n", T->nvec) ;
    #endif

    // construct Tp, Th, Ti, and Tx, summing up duplicates
    GB_cuda_builder_phase5 <<<grid, block1, 0, mystream>>>
        (/* outputs: */ T,
         /* inputs: */  Map, ChunkSum,
            #if GB_BUILD_MATRIX
            JDelta, JDeltaSum,
            #endif
            Key_out, Sx, nvals, nchunks) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (mystream)) ;

    #if 0
    GB_Tp_TYPE *__restrict__ Tp = (GB_Tp_TYPE *) T->p ;
    #if GB_BUILD_MATRIX
    GB_Tj_TYPE *__restrict__ Th = (GB_Tj_TYPE *) T->h ;
    #endif
    for (int k = 0 ; k <= std::min (nvals, T->nvec) ; k++)
    {
        int64_t p = Tp [k] ;
        printf ("Tp [%d] = %ld,     ", k, p) ;
        #if GB_BUILD_MATRIX
        if (k < T->nvec)
        {
            int64_t j = Th [k] ;
            printf ("Th [%d] = %ld", k, j) ;
        }
        #endif
        printf ("\n") ;
    }
    #endif

//  FIXME: use the input stream; remove mystream and mystream3
//  CUDA_OK (cudaStreamDestroy (mystream)) ;
//  CUDA_OK (cudaStreamDestroy (mystream3)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*Thandle) = T ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

