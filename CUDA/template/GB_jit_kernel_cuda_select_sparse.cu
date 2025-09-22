//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_select_sparse_NEW
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
// GB_cuda_select_sparse_phase1_NEW: construct Ak, and first phase for Map
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase1_NEW
(
    // outputs
    GB_Aj_TYPE *Ak,     // size anz, values in range 0 to max # vectors in A
    GB_Ap_TYPE *Map,    // size anz, values in range 0 to anz-1
    int64_t *GlobalSum, // size # threadblocks + 1
    // inputs, not modified
    GrB_Matrix A,
    const void *ythunk
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
    GB_A_NHELD (anz) ;

    // workspace for each chunk
    __shared__ GB_Ap_TYPE Local_Map [CHUNK_SIZE] ;

    //--------------------------------------------------------------------------
    // compute Ak, and each local chunk of Map
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << LOG2_CHUNK_SIZE ;
                 pfirst < anz ;
                 pfirst += gridDim.x << LOG2_CHUNK_SIZE )
    {

        //----------------------------------------------------------------------
        // determine the chunk and its slope
        //----------------------------------------------------------------------

        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, CHUNK_SIZE,
            &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        //----------------------------------------------------------------------
        // find the kth vector that contains each entry pA = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {

            //------------------------------------------------------------------
            // determine the kth vector that contains the pA-th entry
            //------------------------------------------------------------------

            int64_t pA = pfirst + pdelta ;
            int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap, anvec1, kfirst, slope) ;

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
        for ( ; pdelta < CHUNK_SIZE ; pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
        }

        this_thread_block ( ).sync ( ) ;
        // do a cub::BlockScan::InclusiveSum on threadblock's
        // Local_Map [0..CHUNK_SIZE-1],
        // where Local_Map [i] = sum (Local_Map [0:i]), so that
        // Local_Map [0] = 1 if the first entry is kept, 0 otherwise,
        // and Local_Map [CHUNK_SIZE-1] = total # entries kept in this block
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // save the Local_Map in Map [pfirst:plast-1], and in GlobalSum
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            Map [pfirst + pdelta] = Local_Map [pdelta] ;
        }

        // last thread writes the sum of the whole threadblock to global
        // FIXME: which thread should do this work?
        if (threadIdx.x == blockDim.x - 1)
        {
            // FIXME: should the outer loop just iterate over the chunks?
            int64_t chunk_id = pfirst >> LOG2_CHUNK_SIZE ;
            GlobalSum [chunk_id] = Local_Map [CHUNK_SIZE-1] ;
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
    GB_Aj_TYPE *Ak_Keep,    // size cnz+1
    // inputs, not modified
    int64_t *GlobalSum,     // size # threadblocks + 1
    GB_Ap_TYPE *Map,        // size anz
    GB_Aj_TYPE *Ak,         // size anz
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
    GB_A_NHELD (anz) ;

    //--------------------------------------------------------------------------
    // select the entries and copy them into Ci, Cx; construct Ak_Keep
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << LOG2_CHUNK_SIZE ;
                 pfirst < anz ;
                 pfirst += gridDim.x << LOG2_CHUNK_SIZE )
    {

        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, anz) ;
        int64_t my_chunk_size = plast - pfirst ;

        // FIXME: should the outer loop just iterate over the chunks?
        int64_t chunk_id = pfirst >> LOG2_CHUNK_SIZE ;

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;

            // get the position pC in C of the pA-th entry in A
            GB_Ap_TYPE pC  = Map [pA  ] + GlobalSum [chunk_id] ;
            GB_Ap_TYPE pC1 = Map [pA-1] + GlobalSum [chunk_id + ((pdelta == 0) ? -1 : 0)] ;
            if (pC1 < pC)
            {
                // This entry is kept; it appears in a new position pC as compared
                // to the entry Map [pA-1] immediately to its left.  Map contains
                // 1-based indices since it was computed as an inclusive cumsum, so
                // Ci, Cx, and Ak_Keep have been shifted by one above.
                Ci [pC] = Ai [pA] ;
                // Cx [pC] = Ax [pA] ;
                GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
                // save the index of the kA-th vector kA that holds this entry in
                // A, for the new position of this entry in C at pC.
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
    GB_Cj_TYPE *Ck_Delta,   // size cnz
    int64_t *GlobalSum,     // size # threadblocks + 1
    // inputs, not modified
    GB_Aj_TYPE *Ak_Keep,    // size cnz
    int64_t cnz
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ GB_Cj_TYPE Local_Ck_Delta [CHUNK_SIZE] ;

    //--------------------------------------------------------------------------
    // construct Ck_Delta and then cumsum each block
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << LOG2_CHUNK_SIZE ;
                 pfirst < cnz ;
                 pfirst += gridDim.x << LOG2_CHUNK_SIZE )
    {

        //---------------------------------------------------------------------
        // get the part of Ak_Keep and Ck_Delta this threadblock works on
        //---------------------------------------------------------------------

        // this threadblock works on Ak_Keep [pfirst:plast-1] and
        // Ck_Delta [pfirst:plast-1]

        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //---------------------------------------------------------------------
        // determine which entries of Ak_Keep start new vectors in C
        //---------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t pC = pfirst + pdelta ;
            GB_Aj_TYPE kA = Ak_Keep [pC] ;
            Local_Ck_Delta [pdelta] = (Ak_Keep [pC-1] < kA) ;
        }

        // clear the unused part of the Local_Ck_Delta
        for ( ; pdelta < CHUNK_SIZE ; pdelta += blockDim.x)
        {
            Local_Ck_Delta [pdelta] = 0 ;
        }

        this_thread_block ( ).sync ( ) ;
        // do a cub::BlockScan::InclusiveSum on threadblock's
        // Local_Ck_Delta [0..CHUNK_SIZE-1]
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // save the Local_Ck_Delta in Ck_Delta [pfirst:plast-1]
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            Ck_Delta [pfirst + pdelta] = Local_Ck_Delta [pdelta] ;
        }

        // last thread writes the sum of the whole threadblock to global
        // FIXME: which thread should do this work?
        if (threadIdx.x == blockDim.x - 1)
        {
            // FIXME: should the outer loop just iterate over the chunks?
            int64_t chunk_id = pfirst >> LOG2_CHUNK_SIZE ;
            GlobalSum [chunk_id] = Local_Ck_Delta [CHUNK_SIZE-1] ;
        }

        this_thread_block ( ).sync ( ) ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase6_NEW:
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase6_NEW
(
    // outputs
    GrB_Matrix C,           // Cp and Ch are constructed
    // inputs, not modified
    GB_Cj_TYPE *Ck_Delta,   // size cnz + 1
    int64_t *GlobalSum,     // size # threadblocks + 1
    GB_Aj_TYPE *Ak_Keep,    // size cnz + 1
    GrB_Matrix A
)
{

    // Cp and Ch use 1-based indexing below, so decrement them by 1
    GB_Cp_TYPE *Cp = C->p ; Cp-- ;
    GB_Cj_TYPE *Ch = C->h ; Ch-- ;
    int64_t cnz = C->nvals ;

    #if ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif

    //--------------------------------------------------------------------------
    // determine the start of each vector in C
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << LOG2_CHUNK_SIZE ;
                 pfirst < cnz ;
                 pfirst += gridDim.x << LOG2_CHUNK_SIZE )
    {

        int64_t plast = pfirst + CHUNK_SIZE ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        // FIXME: should the outer loop just iterate over the chunks?
        int64_t chunk_id = pfirst >> LOG2_CHUNK_SIZE ;

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {

            int64_t pC = pfirst + pdelta ;

            // compute Ck_Map [pC] and Ck_Map [pC-1]
            GB_Cj_TYPE ck_map_pC  = Ck_Delta [pC  ] + GlobalSum [chunk_id] ;
            GB_Cj_TYPE ck_map_pC1 = Ck_Delta [pC-1] + GlobalSum [chunk_id + ((pdelta == 0) ? -1 : 0)] ;

            if (ck_map_pC != ck_map_pC1)
            {
                // this is the start of a new vector in C
                // note that ck_map_pC is 1-based
                int64_t kA = Ak_keep [pC] ;
                Cp [ck_map_pC] = pC ;
                Ch [ck_map_pC] = GBh_A (Ah, kA) ;
            }
        }
    }

    // finalize the last vector of C
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
    // workspace of size gridsz+4
    void *W_2 = NULL ; size_t W_2_size = 0 ;
    // workspace of size cnz+2, where cnz <= anz
    void *W_3 = NULL ; size_t W_3_size = 0 ;

    GB_A_NHELD (anz) ;          // # of entries in A
    int64_t cnz = 0 ;           // # of entries in C (which is <= anz)

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;        // = min (ceil (nnz(A)/512), 256*(#sms))
    dim3 block (blocksz) ;      // = 512

    //--------------------------------------------------------------------------
    // phase 1: determine which entries of A to keep
    //--------------------------------------------------------------------------

    // This phase constructs Ak [0..anz-1], where Ak [pA] = kA if the pA-th
    // entry is in the kA-th vector of A.  It also is the first phase in
    // constructing Map [0..anz-1], where Map [pA] = pC if the pA-th entry
    // of A is the pC-th entry of C.

    size_t w0 = GB_IMAX (sizeof (GB_Aj_TYPE), sizeof (GB_Cj_TYPE)) ;
    W_0 = (void *) GB_MALLOC_MEMORY (anz+2, w0, &W_0_size) ;
    W_1 = (void *) GB_MALLOC_MEMORY (anz+2, sizeof (Gp_Ap_TYPE), &W_1_size) ;
    W_2 = (void *) GB_MALLOC_MEMORY (gridsz+4, sizeof (int64_t), &W_2_size) ;
    if (W_0 == NULL || W_1 == NULL || W_2 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // use W_0 [1..anz] as workspace for Ak [0..anz-1]
    GB_Aj_TYPE *Ak = (GB_Aj_TYPE *) W_0 + 1 ;

    // use W_1 workspace for Map, and shift by one to define Map [-1] as 0
    GB_Ap_TYPE *Map = ((GB_Ap_TYPE *) W_1) + 1 ;
    Map [-1] = 0 ;

    // GlobalSum [-1 .. #threadblocks] of size (#threadblocks+2)
    int64_t *GlobalSum = (int64_t *) W_2 - 1 ;
    GlobalSum [-1] = 0 ;

    GB_cuda_select_sparse_phase1_NEW <<<grid, block, 0, stream>>>
        (Ak, Map, GlobalSum, A, ythunk) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 2: sum up the entries in each block (on the CPU) and allocate C
    //--------------------------------------------------------------------------

    // overwrite GlobalSum [0..gridsdz] with its cumulative sum
    int64_t cnz = 0 ;
    for (int64_t blockid = 0 ; blockid < gridsz ; blockid++)
    {
        // get the # of entries found by this threadblock
        int64_t s = GlobalSum [blockid] ;
        // overwrite the entry with the cumulative sum, so that the new
        // GlobalSum [blockid] = original GlobalSum [0..blockid-1]
        GlobalSum [blockid] = cnz ;
        cnz += s ;
    }
    GlobalSum [gridsz] = cnz ;

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

    // use W_3 as workspace for Ak_Keep, and shift Ak_Keep by one
    GB_Aj_TYPE *Ak_Keep = ((GB_Aj_TYPE *) W_3) + 1 ;
    Ak_Keep [-1] = -1 ;

    //--------------------------------------------------------------------------
    // phase 3: finalize the Map and construct Ci, Cx, and Ak_keep
    //--------------------------------------------------------------------------

    GB_cuda_select_sparse_phase3_NEW <<<grid, block, 0, stream>>>
        (C, Ak_Keep, GlobalSum, Map, Ak, A, ythunk) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ak (in W_0) no longer needed; reused below for Ck_Delta
    // Map (in W_1) no longer needed

    //--------------------------------------------------------------------------
    // phase 4: construct Ck_Delta and its local cumulative sum
    //--------------------------------------------------------------------------

    // using W_0 as workspace for Ck_Delta
    GB_Cj_TYPE *Ck_Delta = ((GB_Cj_TYPE *) W_0) + 1 ;
    Ck_Delta [-1] = 0 ;

    // Ck_Delta [pC] = 1 if the pC-th entry is the first in its vector of C, or
    // 0 otherwise.  Then each threadblock computes the cumulative sum of its
    // part of Ck_Delta.

    GB_cuda_select_sparse_phase4_NEW <<<grid, block, 0, stream>>>
        (Ck_Delta, GlobalSum, Ak_Keep) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ak_Keep (in W_3) no longer needed

    //--------------------------------------------------------------------------
    // phase 5: construct global cumsum of Ck_Delta on the CPU; allocate Cp,Ch
    //--------------------------------------------------------------------------

    // overwrite GlobalSum [0..gridsdz] with its cumulative sum
    int64_t cnvec = 0 ;
    for (int64_t blockid = 0 ; blockid < gridsz ; blockid++)
    {
        // get the # of entries found by this threadblock
        int64_t s = GlobalSum [blockid] ;
        // overwrite the entry with the cumulative sum, so that the new
        // GlobalSum [blockid] = original GlobalSum [0..blockid-1]
        GlobalSum [blockid] = cnvec ;
        cnvec += s ;
    }
    GlobalSum [gridsz] = cnvec ;

    // The caller has already allocated C->p, C->h for
    // a user-returnable empty hypersparse matrix.
    // Free them here before updating.
    GB_FREE_MEMORY (&(C->p), C->p_size) ;
    GB_FREE_MEMORY (&(C->h), C->h_size) ;

    // Allocate Cp, Ch, finalize matrix
    C->plen = cnvec ;
    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;
    C->p = (GB_Cp_TYPE *) GB_MALLOC_MEMORY (C->plen+1, sizeof (GB_Cp_TYPE), &(C->p_size)) ;
    C->h = (GB_Cj_TYPE *) GB_MALLOC_MEMORY (C->plen, sizeof (GB_Cj_TYPE), &(C->h_size)) ;
    if (C->p == NULL || C->h == NULL)
    {
        // The contents of C will be freed with GB_phybix_free()
        // in the caller (GB_cuda_select_sparse()) upon returning
        // an error.
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // phase 6: construct Cp and Ch
    //--------------------------------------------------------------------------

    GB_cuda_select_sparse_phase6_NEW <<<grid, block, 0, stream>>>
        (C, Ck_Delta, GlobalSum, A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

