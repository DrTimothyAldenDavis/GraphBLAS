//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_select_sparse_NEW
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

using namespace cooperative_groups ;

#include "GB_cuda_ek_slice.cuh"
#include "GB_cuda_cumsum.cuh"

#define GB_FREE_WORKSPACE                               \
{                                                       \
    GB_FREE_MEMORY (&Ak, Ak_size) ;                     \
    GB_FREE_MEMORY (&W_1, W_1_size) ;                   \
    GB_FREE_MEMORY (&W_2, W_2_size) ;                   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE ;

#define chunk_size 1024
#define log2_chunk_size 10

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1_NEW: construct Ak, and first phase for Map
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase1_NEW
(
    // outputs
    GB_Ap_TYPE *Ak,     // size nvals(A)
    GB_Ap_TYPE *Map,    // size nvals(A)
    // inputs, not modified
    GrB_Matrix A,
    const void *ythunk
)
{

    //--------------------------------------------------------------------------
    // get A and ythunk
    //--------------------------------------------------------------------------

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
    GB_A_NHELD (anz) ;

    // workspace for each threadblock
    __shared__ GB_Ap_TYPE Local_Map [chunk_size+1] ;

    //--------------------------------------------------------------------------
    // compute Ak, and each local block of Map
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < anz ;
                 pfirst += gridDim.x << log2_chunk_size )
    {

        //----------------------------------------------------------------------
        // determine the chunk for this threadblock and its slope
        //----------------------------------------------------------------------

        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, chunk_size,
            &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        //----------------------------------------------------------------------
        // find the kth vector that contains each entry pA = pfirst:plast-1
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {

            //------------------------------------------------------------------
            // determine the kth vector that contains the pA-th entry
            //------------------------------------------------------------------

            int64_t pA ;    // pA = pfirst + pdelta
            int64_t k = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (&pA, pdelta,
                pfirst, Ap, anvec1, kfirst, slope) ;

            //------------------------------------------------------------------
            // save the vector index k, and determine if this entry is kept 
            //------------------------------------------------------------------

            Ak [pA] = k ;
            #if ( GB_DEPENDS_ON_J )
            int64_t j = GBh_A (Ah, k) ;
            #endif
            #if ( GB_DEPENDS_ON_I )
            int64_t i = Ai [pA] ;
            #endif
            // keep = fselect (A (i,j)), 1 if A(i,j) is kept, else 0
            GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
            Local_Map [pdelta] = keep ;
        }

        this_thread_block ( ).sync ( ) ;
        // HERE: do a cub::BlockScan on threadblock's Local_Map
        // where Local_Map [i] = sum (Local_Map [0:i]), so that
        // Local_Map [0] = 1 if the first entry is kept, 0 otherwise,
        // and Local_Map [my_chunk_size-1] = total # entries kept in this block
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // save the Local_Map in Map [pfirst:plast-1]
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            Map [pfirst + pdelta] = Local_Map [pdelta] ;
        }

        this_thread_block ( ).sync ( ) ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase3_NEW: finalize Map and construct Ci, Cx, Ak_Keep
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase3_NEW
(
    // input/outputs
    GrB_Matrix C,           // construct C->i and C->x
    GB_Ap_TYPE *Ak_Keep,    // size nvals(C)
    // inputs, not modified
    GB_Ap_TYPE *GlobalSum,  // size # threadblocks
    GB_Ap_TYPE *Map,        // size nvals(A)
    GB_Ap_TYPE *Ak,         // size nvals(A)
    GrB_Matrix A,
    const void *ythunk
)
{

    //--------------------------------------------------------------------------
    // get C->i and C->x, shifting down by one to account for 1-based Map
    //--------------------------------------------------------------------------

    // in this method, the index pC = Map [pA] is 1-based, so decrement Ci, Cx,
    // and Ak_Keep to account for this.

    GB_Ci_TYPE *Ci = C->i ; Ci-- ;
    #if !GB_ISO_SELECT
    GB_C_TYPE  *Cx = C->x ; Cx-- ;
    #endif
    Ak_Keep-- ;

    //--------------------------------------------------------------------------
    // select the entries and copy them into Ci, Cx; construct Ak_Keep
    //--------------------------------------------------------------------------

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pA = tid ; pA < anz ; pA += nthreads)
    {
        // get the position pC in C of the pA-th entry in A
        GB_Ap_TYPE pC   = Map [pA] + GlobalSum [this thread block] ;
        GB_Ap_TYPE pC_1 = Map [pA-1] + GlobalSum [thread block with pA-1] ;

        if (Map [pA-1] < pC)
        {
            // This entry is kept; it appears in a new position pC as compared
            // to the entry Map [pA-1] immediately to its left.  Map contains
            // 1-based indices since it was computed as an inclusive cumsum, so
            // Ci, Cx, and Ak_Keep have been shifted by one above.
            Ci [pC] = Ai [pA] ;
            // Cx [pC] = Ax [pA] ;
            GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
            // save the name of the vector kA that holds this entry in A,
            // for the new position of this entry in C at pC.
            Ak_Keep [pC] = Ak [pA] ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase4_NEW: construct each block of Ck_Delta
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase4_NEW
(
    // outputs
    GB_Cp_TYPE *Ck_Delta,   // size nvals(C)
    // inputs, not modified
    GB_Ap_TYPE *Ak_Keep,    // size nvals(C)
    int64_t cnz             // cnz = nvals(C)
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ GB_Cp_TYPE Local_Ck_Delta [chunk_size+1] ;

    //--------------------------------------------------------------------------
    // construct Ck_Delta and then cumsum each block
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < cnz ;
                 pfirst += gridDim.x << log2_chunk_size )
    {

        //---------------------------------------------------------------------
        // get the part of Ak_Keep and Ck_Delta this threadblock works on
        //---------------------------------------------------------------------

        // this threadblock works on Ak_Keep [pfirst:plast-1] and
        // Ck_Delta [pfirst:plast-1]

        int64_t plast = pfirst + chunk_size ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //---------------------------------------------------------------------
        // determine which entries of Ak_Keep start new vectors in C
        //---------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pC = pfirst + pdelta ;
            GB_Ap_TYPE kA = Ak_Keep [pC] ;
            Local_Ck_Delta [pdelta] = (Ak_Keep [pC-1] < kA) ;
        }

        this_thread_block ( ).sync ( ) ;
        // HERE: do a cub::BlockScan on threadblock's Local_Ck_Delta
        this_thread_block ( ).sync ( ) ;
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

    // workspaces of size nvals(A)
    GB_Ap_TYPE *Ak = NULL ;         size_t Ak_size = 0 ;
    void *W_1 = NULL ;              size_t W_1_size = 0 ;
    // workspace of size nvals(C)
    void *W_2 = NULL ;              size_t W_2_size = 0 ;

    int64_t cnz = 0 ;
    GB_A_NHELD (anz) ;

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;        // = min (ceil (nnz(A)/512), 256*(#sms))
    dim3 block (blocksz) ;      // = 512

    //--------------------------------------------------------------------------
    // phase 1: determine which entries of A to keep
    //--------------------------------------------------------------------------

    // This phase constructs Ak [0..anz-1], where Ak [pA] = k if the pA-th
    // entry is in the kth vector of A.  It also is the first phase in
    // constructing Map [0..anz-1], where Map [pA] = pC if the pA-th entry
    // of A is the pC-th entry of C.

    Ak = (GB_Ap_TYPE *) GB_MALLOC_MEMORY (anz+1, sizeof (GB_Ap_TYPE),
        &Ak_size) ;
    size_t w1 = GB_IMAX (sizeof (GB_Ap_TYPE), sizeof (GB_Cp_TYPE)) ;
    W_1 = (void *) GB_MALLOC_MEMORY (anz+1, w1, &W_1_size) ;
    if (Ak == NULL || W_1 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one: to define Map [-1] as 0
    GB_Ap_TYPE *Map = ((GB_Ap_TYPE *) W_1) + 1 ;
    Map [-1] = 0 ;

    GB_cuda_select_sparse_phase1_NEW <<<grid, block, 0, stream>>>
        (Ak, Map, A, ythunk) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 2: sum up the entries in each block (on the CPU) and allocate C
    //--------------------------------------------------------------------------

    // declare GlobalSum [0.. #threadblocks-1] ... malloc it?

    int64_t blockid = 0 ;
    int64_t c = 0 ;
    for (int64_t plast = blocksz - 1 ; plast < anz - 1 ; plast += blocksz)
    {
        // get the # of entries found by this threadblock
        c += Map [plast] ;
        GlobalSum [blockid++] = c ;
    }

    // add the # of entries in the last block
    c += Map [anz-1] ;
    GlobalSum [blockid] = c ;

    int64_t cnz = c ;               // total # of entries kept, for C

    GB_OK (GB_bix_alloc (C, cnz, GxB_HYPERSPARSE, false, true, GB_ISO_SELECT)) ;
    C->nvals = cnz ;

    if (cnz == 0)
    {
        // C is empty; nothing more to do
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    // allocate workspace
    W_2 = (GB_Ap_TYPE *) GB_MALLOC_MEMORY (cnz + 1, sizeof (GB_Ap_TYPE),
        &W_2_size) ;
    if (W_2 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift Ak_Keep by one
    GB_Ap_TYPE *Ak_Keep = ((GB_Ap_TYPE *) W_2) + 1 ;
    Ak_Keep [-1] = 0 ;

    //--------------------------------------------------------------------------
    // phase 3: finalize the Map and construct Ci, Cx, and Ak_keep
    //--------------------------------------------------------------------------

    GB_cuda_select_sparse_phase3_NEW <<<grid, block, 0, stream>>>
        (C, Ak_Keep, Map, Ak, A, ythunk) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Map no longer needed

    //--------------------------------------------------------------------------
    // phase 4: construct Ck_Delta and its local cumulative sum
    //--------------------------------------------------------------------------

    GB_Cp_TYPE *Ck_Delta = ((GB_Cp_TYPE *) W_1) + 1 ;
    Ck_Delta [-1] = 0 ;

    // Ck_Delta [pC] = 1 if the pC-th entry is the first in its vector of C, or
    // 0 otherwise.  Next, the each threadblock computes the cumulative sum of
    // its part of Ck_Delta.

    GB_cuda_select_sparse_phase4_NEW <<<grid, block, 0, stream>>>
        (Ck_Delta, Ak_Keep) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ak_Keep no longer needed

    //--------------------------------------------------------------------------
    // phase 5: construct global cumsum of Ck_Delta on the CPU
    //--------------------------------------------------------------------------

    // declare GlobalSum [0.. #threadblocks-1]

    blockid = 0 ;
    c = 0 ;
    for (int64_t plast = blocksz - 1 ; plast < cnz - 1 ; plast += blocksz)
    {
        // get the # of entries found by this threadblock
        c += Ck_Delta [plast] ;
        GlobalSum [blockid++] = c ;
    }

    // add the # of entries in the last block
    c += Ck_Delta [cnz-1] ;
    GlobalSum [blockid] = c ;

    //--------------------------------------------------------------------------
    // phase 6: construct Ck_Map
    //--------------------------------------------------------------------------



    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

