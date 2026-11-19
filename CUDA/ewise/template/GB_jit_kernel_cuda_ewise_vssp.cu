//------------------------------------------------------------------------------
// GraphBLAS/CUDA/ewise/template/GB_jit_kernel_cuda_ewise_vssp.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Binary-search variant of the eWise kernel, for the skewed case where one
// operand is far sparser than the other (nnz(A) << nnz(B)).  A is the SCANNED
// (sparse) operand; B is the SEARCHED (dense) operand.  For each entry of A we
// recover its column j (getk), locate B(:,j), and binary-search its row in B.

// It serves the two set operations whose output is a subset of A:
//   GB_SET_INTERSECTION : eWiseMult, and C<M>=A  (emit A(i,j) if B(i,j) found)
//   GB_SET_DIFFERENCE   : C<!M>=A, B is mask M   (emit A(i,j) if NOT found)
// It is not used for GB_SET_UNION: there |C| ~ |B|, so a search saves nothing.

// This kernel mirrors GB_jit_kernel_cuda_ewise_sparse.cu (the merge kernel).
// The phase numbering matches it (phase1, phase3, phase5, phase6, phase7); there
// is no phase2 since the tasks are equal-size slices of A.  The only difference
// from the merge kernel is the inner method: a binary search into B(:,j) rather
// than a merge-path over shared memory.

//   phase 1 : build the A-slice array so getk can recover each A column
//   phase 3 : one block per A-slice; count surviving entries -> |C_t|
//   phase 4 : (host) exclusive cumsum of |C_t|, allocate C
//   phase 5 : same search; pack survivors at Task_Cstart[t] via a BlockScan
//   phase 6 : find leading entries of each column of C
//   phase 7 : construct Cp and Ch

// The "finger": each thread walks a CONTIGUOUS run of A, so a column's rows are
// seen in ascending order.  B(:,j) is looked up once per column and each binary
// search starts at the previous search's position -- shorter searches, and
// B(:,j) stays hot in cache (this removes the deep-column pathology).

#include "template/GB_cuda_ek_slice.cuh"
#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_threadblock_sum_uint64.cuh"

// phase1 + getk: size of one ek_slice slice
#define CHUNKSIZE1 256
#define LOG2_CHUNKSIZE1 8

// task size: # of A entries handled by one threadblock (equal slices of A).
// Kept small so there are many tasks (occupancy); the finger in phase3/phase5
// still keeps each thread's few contiguous entries in the same B column
// (cache-hot) and shortens each search.  Tuned on a Tesla T4.
#define CHUNKSIZE2 512
#define LOG2_CHUNKSIZE2 9

// block widths
#define BLOCKDIM1 32
#define BLOCKDIM3 256               // phase3 (count)
#define BLOCKDIM5 BLOCKDIM3         // phase5 (fill); cub::BlockScan uses this

// phase6 and phase7 (same geometry as the builder kernel)
#define BLOCKDIM6 128
#define CHUNKSIZE6 256
#define LOG2_CHUNKSIZE6 8

#define CHUNK_DELTA_MASK1 (CHUNKSIZE1 - 1)

#define GB_FREE_WORKSPACE                                     \
{                                                             \
    GB_FREE_MEMORY (&A_slice_kfirst, A_slice_kfirst_mem) ;    \
    GB_FREE_MEMORY (&Task_Cstart, Task_Cstart_mem) ;          \
    GB_FREE_MEMORY (&W_0, W_0_mem) ;                          \
    GB_FREE_MEMORY (&W_6, W_6_mem) ;                          \
    GB_FREE_MEMORY (&W_7, W_7_mem) ;                          \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL     \
{                       \
    GB_FREE_WORKSPACE ; \
}

//------------------------------------------------------------------------------
// getk: find the vector k that owns entry p  (same as the merge kernel)
//------------------------------------------------------------------------------

template <typename T> __device__ __inline__ int64_t getk
(
    const int64_t p,              // position of the entry
    const int64_t *Slice_kfirst,  // the small array, size nchunks+1
    const T *Ap,                  // column pointers, size anvec+1
    const int64_t anvec1          // anvec-1
)
{
    int64_t slice  = p >> LOG2_CHUNKSIZE1 ;
    int64_t pdelta = p & CHUNK_DELTA_MASK1 ;
    int64_t kfirst = Slice_kfirst [slice] ;
    int64_t klast  = Slice_kfirst [slice+1] ;
    float   slope  = ((float) (klast - kfirst + 1)) / ((float) CHUNKSIZE1) ;

    return (GB_cuda_ek_slice_entry<T> (p, pdelta, Ap, anvec1, kfirst, slope)) ;
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_vssp_phase1: build the A-slice array for getk  (A only)
//------------------------------------------------------------------------------

__global__ void GB_cuda_ewise_vssp_phase1
(
    int64_t *A_slice_kfirst,        // out: size A_slice_nchunks+1
    int64_t A_slice_nchunks,        // in
    const GrB_Matrix A
)
{
    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;

    // one thread per slice of A; k never decreases, so each search is short
    int64_t k = 0 ;
    for (int64_t slice = blockIdx.x * blockDim.x + threadIdx.x ;
            slice < A_slice_nchunks ;
            slice += blockDim.x * gridDim.x)
    {
        int64_t pfirst = slice << LOG2_CHUNKSIZE1 ;
        GB_cuda_ek_slice_search<GB_Ap_TYPE> (&k, Ap, anvec, pfirst) ;
        A_slice_kfirst [slice] = k ;
    }

    // sentinel for the last slice
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        A_slice_kfirst [A_slice_nchunks] = anvec ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_vssp_phase3: compute the size of C for each task  (|C_t|)
//------------------------------------------------------------------------------

__global__ void GB_cuda_ewise_vssp_phase3
(
    // outputs:
    int64_t *Task_Cstart,       // size ntasks+1; Task_Cstart [t] = |C_t|
    // inputs:
    const int64_t ntasks,
    const int64_t anz,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *__restrict__ A_slice_kfirst
)
{
    // inputs derived from A  (as in the merge kernel)
    const GB_Ap_TYPE *Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *Ah = (GB_Aj_TYPE *) A->h ;    // NULL if sparse
    const GB_Ai_TYPE *Ai = (GB_Ai_TYPE *) A->i ;
    const int64_t anvec1 = A->nvec - 1 ;

    // inputs derived from B  (plus hyper-hash fields, as in dot3 vssp)
    const GB_Bp_TYPE *Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bi_TYPE *Bi = (GB_Bi_TYPE *) B->i ;
    #if GB_B_IS_HYPER
    const GB_Bj_TYPE *Bh = (GB_Bj_TYPE *) B->h ;
    const int64_t bnvec = B->nvec ;
    const void *B_Yp = (void *) ((B->Y == NULL) ? NULL : B->Y->p) ;
    const void *B_Yi = (void *) ((B->Y == NULL) ? NULL : B->Y->i) ;
    const void *B_Yx = (void *) ((B->Y == NULL) ? NULL : B->Y->x) ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;
    #endif

    // one threadblock per task; a task is an equal-size slice of A's entries
    for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
    {
        int64_t pA     = (int64_t) t << LOG2_CHUNKSIZE2 ;
        int64_t pA_end = GB_IMIN (pA + CHUNKSIZE2, anz) ;
        uint64_t AB_intersection = 0 ;

        // each thread walks a CONTIGUOUS sub-run of this task, so consecutive A
        // entries in the same column are seen in ascending row order.  B(:,j) is
        // looked up once per column, and each binary search starts at the prior
        // search's position (the "finger"): the rows ascend, so the match can
        // only move to the right.
        int n      = (int) (pA_end - pA) ;
        int run    = GB_ICEIL (n, blockDim.x) ;
        int qstart = threadIdx.x * run ;
        int qend   = GB_IMIN (qstart + run, n) ;

        int64_t last_j = -1 ;           // column currently loaded
        int64_t pB = 0, pB_end = 0 ;    // B(:,last_j) range
        int64_t lo = 0 ;                // finger: lower bound of the next search

        for (int q = qstart ; q < qend ; q++)
        {
            int64_t p  = pA + q ;
            int64_t iA = Ai [p] ;
            int64_t k  = getk<GB_Ap_TYPE> (p, A_slice_kfirst, Ap, anvec1) ;
            int64_t jA = GB_A_IS_HYPER ? Ah [k] : (int64_t) k ;

            // on a column change, look up B(:,jA) and reset the finger
            if (jA != last_j)
            {
                #if GB_B_IS_HYPER
                GB_hyper_hash_lookup (GB_Bp_IS_32, GB_Bj_IS_32,
                    Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
                    jA, &pB, &pB_end) ;
                #else
                pB     = Bp [jA] ;
                pB_end = Bp [jA+1] ;
                #endif
                last_j = jA ;
                lo = pB ;
            }

            // fingered binary search for row iA in Bi [lo : pB_end-1]
            bool found = false ;
            if (lo < pB_end)
            {
                int64_t pleft = lo ;
                int64_t pright = pB_end - 1 ;
                found = GB_binary_search (iA, Bi, GB_Bi_IS_32, &pleft, &pright) ;
                lo = pleft ;            // carry the finger forward
            }

            #if GB_SET_INTERSECTION
            AB_intersection += (found) ;
            #elif GB_SET_DIFFERENCE
            AB_intersection += (!found) ;
            #endif
        }

        // reduce the block's counts into thread 0
        AB_intersection = GB_cuda_threadblock_sum_uint64 (AB_intersection) ;
        if (threadIdx.x == 0)
        {
            Task_Cstart [t] = AB_intersection ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_vssp_phase5: compute pattern and values of C (coordinate form)
//------------------------------------------------------------------------------

__global__ void GB_cuda_ewise_vssp_phase5
(
    // outputs:
    GrB_Matrix C,
    GB_Cj_TYPE *Cj,                             // size cnz+1, Cj [-1..cnz-1]
    // inputs:
    const int64_t *__restrict__ Task_Cstart,    // size ntasks+1 (output bases)
    const int64_t ntasks,
    const int64_t anz,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *__restrict__ A_slice_kfirst,
    const void *theta
)
{
    // inputs derived from A  (as in the merge kernel)
    const GB_Ap_TYPE *Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *Ah = (GB_Aj_TYPE *) A->h ;    // NULL if sparse
    const GB_Ai_TYPE *Ai = (GB_Ai_TYPE *) A->i ;
    const int64_t anvec1 = A->nvec - 1 ;

    // inputs derived from B  (plus hyper-hash fields, as in dot3 vssp)
    const GB_Bp_TYPE *Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bi_TYPE *Bi = (GB_Bi_TYPE *) B->i ;
    #if GB_B_IS_HYPER
    const GB_Bj_TYPE *Bh = (GB_Bj_TYPE *) B->h ;
    const int64_t bnvec = B->nvec ;
    const void *B_Yp = (void *) ((B->Y == NULL) ? NULL : B->Y->p) ;
    const void *B_Yi = (void *) ((B->Y == NULL) ? NULL : B->Y->i) ;
    const void *B_Yx = (void *) ((B->Y == NULL) ? NULL : B->Y->x) ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;
    #endif

    #if !GB_C_ISO
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #if !GB_IS_MASK_ASSIGN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *) B->x ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) C->x ;
    #endif
          GB_Ci_TYPE *__restrict__ Ci = (GB_Ci_TYPE *) C->i ;

    // cub::BlockScan workspace for the exclusive sum of per-thread counts
    using BlockScan = cub::BlockScan <uint16_t, BLOCKDIM5,
        cub::BLOCK_SCAN_WARP_SCANS> ;
    __shared__ typename BlockScan::TempStorage W ;

    for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
    {
        int64_t pA     = (int64_t) t << LOG2_CHUNKSIZE2 ;
        int64_t pA_end = GB_IMIN (pA + CHUNKSIZE2, anz) ;
        int64_t pC     = Task_Cstart [t] ;      // this task's output base

        // each thread walks a CONTIGUOUS sub-run (see phase3).  Pass 1 counts
        // this thread's survivors with the finger; one BlockScan gives each
        // thread its output base; pass 2 re-walks and writes the survivors.
        int n      = (int) (pA_end - pA) ;
        int run    = GB_ICEIL (n, blockDim.x) ;
        int qstart = threadIdx.x * run ;
        int qend   = GB_IMIN (qstart + run, n) ;

        //----------------------------------------------------------------------
        // pass 1: count the survivors in my run (fingered)
        //----------------------------------------------------------------------

        uint16_t pc = 0 ;
        {
            int64_t last_j = -1, pB = 0, pB_end = 0, lo = 0 ;
            for (int q = qstart ; q < qend ; q++)
            {
                int64_t p  = pA + q ;
                int64_t iA = Ai [p] ;
                int64_t k  = getk<GB_Ap_TYPE> (p, A_slice_kfirst, Ap, anvec1) ;
                int64_t jA = GB_A_IS_HYPER ? Ah [k] : (int64_t) k ;
                if (jA != last_j)
                {
                    #if GB_B_IS_HYPER
                    GB_hyper_hash_lookup (GB_Bp_IS_32, GB_Bj_IS_32,
                        Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
                        jA, &pB, &pB_end) ;
                    #else
                    pB     = Bp [jA] ;
                    pB_end = Bp [jA+1] ;
                    #endif
                    last_j = jA ;
                    lo = pB ;
                }
                bool found = false ;
                if (lo < pB_end)
                {
                    int64_t pleft = lo, pright = pB_end - 1 ;
                    found = GB_binary_search (iA, Bi, GB_Bi_IS_32,
                        &pleft, &pright) ;
                    lo = pleft ;
                }
                #if GB_SET_INTERSECTION
                pc += (found) ;
                #elif GB_SET_DIFFERENCE
                pc += (!found) ;
                #endif
            }
        }

        //----------------------------------------------------------------------
        // compaction: my output base within the task
        //----------------------------------------------------------------------

        uint16_t pc_start ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W).ExclusiveSum (pc, pc_start) ;
        this_thread_block ( ).sync ( ) ;

        //----------------------------------------------------------------------
        // pass 2: re-walk and write my survivors starting at pC + pc_start
        //----------------------------------------------------------------------

        {
            int64_t out = pC + pc_start ;
            int64_t last_j = -1, pB = 0, pB_end = 0, lo = 0 ;
            for (int q = qstart ; q < qend ; q++)
            {
                int64_t p  = pA + q ;
                int64_t iA = Ai [p] ;
                int64_t k  = getk<GB_Ap_TYPE> (p, A_slice_kfirst, Ap, anvec1) ;
                int64_t jA = GB_A_IS_HYPER ? Ah [k] : (int64_t) k ;
                if (jA != last_j)
                {
                    #if GB_B_IS_HYPER
                    GB_hyper_hash_lookup (GB_Bp_IS_32, GB_Bj_IS_32,
                        Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
                        jA, &pB, &pB_end) ;
                    #else
                    pB     = Bp [jA] ;
                    pB_end = Bp [jA+1] ;
                    #endif
                    last_j = jA ;
                    lo = pB ;
                }
                int64_t pb = 0 ;
                bool found = false ;
                if (lo < pB_end)
                {
                    int64_t pleft = lo, pright = pB_end - 1 ;
                    found = GB_binary_search (iA, Bi, GB_Bi_IS_32,
                        &pleft, &pright) ;
                    lo = pleft ;
                    pb = pleft ;
                }
                #if GB_SET_INTERSECTION
                if (found)
                {
                    // C(i,j) = A(i,j) .* B(i,j)  (collapses to A(i,j) for C<M>=A)
                    Ci [out] = iA ;
                    Cj [out] = jA ;
                    GB_EWISE_AIJ_OP_BIJ (Cx, out, Ax, p, GB_A_ISO,
                        Bx, pb, GB_B_ISO, iA, jA) ;
                    out++ ;
                }
                #elif GB_SET_DIFFERENCE
                if (!found)
                {
                    // C(i,j) = A(i,j)  (C<!M>=A; B is the mask, beta unused)
                    Ci [out] = iA ;
                    Cj [out] = jA ;
                    GB_EWISE_AIJ_OP_BETA (Cx, out, Ax, p, GB_A_ISO, 0, iA, jA) ;
                    out++ ;
                }
                #endif
            }
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_vssp_phase6: find leading entries of each column of C
//------------------------------------------------------------------------------

#include "template/GB_cuda_construct_JDelta.cuh"

#define ITEMS_PER_THREAD6 ( CHUNKSIZE6 / BLOCKDIM6 )

__global__ void GB_cuda_ewise_vssp_phase6
(
    // outputs:
    uint16_t *JDelta,           // size cnz+1, in JDelta [-1..cnz-1]
    GB_Cp_TYPE *JDeltaSum,      // size nchunks_in_C+2
    // inputs, not modified, except for Cj [-1] sentinel value:
    GB_Cj_TYPE *Cj,             // size cnz+1: Cj [-1..cnz-1]
    int64_t cnz,
    int64_t nchunks_in_C
)
{
    auto unload_Cj = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        return (Cj [p]) ;
    } ;

    GB_cuda_construct_JDelta
    <
        GB_Cp_TYPE,         // type of JDeltaSum
        GB_Cj_TYPE,         // type of Cj
        CHUNKSIZE6,         // size of each chunk
        LOG2_CHUNKSIZE6,    // log2 (chunksize)
        BLOCKDIM6,          // blockdim of kernel launch
        ITEMS_PER_THREAD6   // # of items per thread (chunksize/blockdim)
    >
        (JDelta, JDeltaSum, Cj, cnz, nchunks_in_C, unload_Cj) ;
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_vssp_phase7: construct Cp and Ch
//------------------------------------------------------------------------------

// This phase is skipped if C->vdim is 1.

#include "template/GB_cuda_construct_Cphix.cuh"

__global__ void GB_cuda_ewise_vssp_phase7
(
    // outputs:
    GrB_Matrix C,
    // inputs, not modified:
    uint16_t *JDelta,       // size nvals+1, in JDelta [-1..nvals-1]
    GB_Cp_TYPE *JDeltaSum,  // size nchunks+1
    GB_Cj_TYPE *Cj,         // size nvals+1: Cj [-1 ... nvals-1]
    int64_t nvals,
    int64_t nchunks
)
{
    auto unload_Ci = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        return (0) ;    // unused
    } ;

    auto unload_Cj = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        return (Cj [p]) ;
    } ;

    auto unload_Cx = [](void *Sx, int64_t p)
    {
        return (0) ;    // unused
    } ;

    GB_cuda_construct_Cphix
    <
        GB_Cp_TYPE,             // type of C->p
        GB_Cj_TYPE,             // type of C->h
        GB_Ci_TYPE,             // type of C->i
        GB_Cj_TYPE,             // type of Cj workspace
        void,                   // type of Cx, not used
        void,                   // type of Sx, not used
        CHUNKSIZE6,             // chunksize for work done by a threadblock
        LOG2_CHUNKSIZE6,        // log2 (chunksize)
        true,                   // C is a matrix; construct C->h
        false,                  // C->i is not constructed
        false                   // C->x is not constructed
    >
        (C, JDelta, JDeltaSum, Cj, NULL, nvals, nchunks,
            unload_Ci, unload_Cj, unload_Cx) ;
}

//------------------------------------------------------------------------------
// cuda_ewise_vssp, host method (issues all phase launches)
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_bix_alloc) ;

    // alpha_scalar_in and beta_scalar_in are unused (no set union here)

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int data_arena = GxB_NARENAS + device ;
    uint64_t mem = GB_mem (data_arena, 0) ;

    int64_t *A_slice_kfirst = NULL ; uint64_t A_slice_kfirst_mem = mem ;
    int64_t *Task_Cstart    = NULL ; uint64_t Task_Cstart_mem    = mem ;

    void *W_0 = NULL ; uint64_t W_0_mem = mem ;    // size cnz+1: Cj
    void *W_6 = NULL ; uint64_t W_6_mem = mem ;    // size nvals+1: JDelta
    void *W_7 = NULL ; uint64_t W_7_mem = mem ;    // size nchunks+2: JDeltaSum

    GB_A_NHELD (anz) ;          // # of entries in the scanned operand A

    int64_t A_slice_nchunks = GB_ICEIL (anz, CHUNKSIZE1) ;

    // tasks are equal-size slices of A: work ~ |A|, so they balance for free
    int64_t ntasks = GB_ICEIL (anz, CHUNKSIZE2) ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block1 (BLOCKDIM1) ;
    dim3 block3 (BLOCKDIM3) ;
    dim3 block5 (BLOCKDIM5) ;
    dim3 block6 (BLOCKDIM6) ;

    //--------------------------------------------------------------------------
    // phase 1: build the A-slice array for getk  (B needs none)
    //--------------------------------------------------------------------------

    A_slice_kfirst = (int64_t *) GB_MALLOC_MEMORY (A_slice_nchunks + 1,
        sizeof (int64_t), &A_slice_kfirst_mem) ;
    Task_Cstart = (int64_t *) GB_MALLOC_MEMORY (ntasks + 1,
        sizeof (int64_t), &Task_Cstart_mem) ;
    if (A_slice_kfirst == NULL || Task_Cstart == NULL)
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 1: phase1 (A only)
    GB_cuda_ewise_vssp_phase1 <<<grid, block1, 0, stream>>>
        (A_slice_kfirst, A_slice_nchunks, A) ;
    CUDA_OK (cudaGetLastError ()) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 3: compute the size of C for each task  (|C_t|)
    //--------------------------------------------------------------------------

    // KERNEL LAUNCH 2: phase3
    GB_cuda_ewise_vssp_phase3 <<<grid, block3, 0, stream>>>
        (Task_Cstart, ntasks, anz, A, B, A_slice_kfirst) ;
    CUDA_OK (cudaGetLastError ()) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 4 (CPU): exclusive cumsum of |C_t|, then allocate C
    //--------------------------------------------------------------------------

    int64_t cnz = 0 ;
    for (int64_t t = 0 ; t < ntasks ; t++)
    {
        int64_t s = Task_Cstart [t] ;
        Task_Cstart [t] = cnz ;     // output base of task t
        cnz += s ;
    }
    Task_Cstart [ntasks] = cnz ;

    // allocate C->[bix]
    GB_OK (GB_bix_alloc (C, cnz, (C->vdim == 1) ? GxB_SPARSE : GxB_HYPERSPARSE,
        /* bitmap_calloc: */ false, /* numeric: */ true, GB_C_ISO)) ;
    C->nvals = cnz ;
    C->jumbled = false ;

    //--------------------------------------------------------------------------
    // phase 5: compute the pattern and values of C in coordinate form
    //--------------------------------------------------------------------------

    // Cj workspace, with the Cj [-1] sentinel value
    W_0 = GB_MALLOC_MEMORY (cnz + 1, sizeof (GB_Cj_TYPE), &W_0_mem) ;
    GB_Cj_TYPE *Cj = ((GB_Cj_TYPE *) W_0) + 1 ;
    if (W_0 == NULL)
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 3: phase5
    GB_cuda_ewise_vssp_phase5 <<<grid, block5, 0, stream>>>
        (C, Cj, Task_Cstart, ntasks, anz, A, B, A_slice_kfirst, theta) ;
    CUDA_OK (cudaGetLastError ()) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 6: find leading entries of each column of C
    //--------------------------------------------------------------------------

    uint16_t *JDelta = NULL ;
    GB_Cp_TYPE *JDeltaSum = NULL ;
    int64_t nchunks_in_C = GB_ICEIL (cnz, CHUNKSIZE6) ;

    if (C->vdim != 1)
    {
        W_6 = GB_MALLOC_MEMORY (cnz+1+CHUNKSIZE6, sizeof (uint16_t), &W_6_mem) ;
        W_7 = GB_MALLOC_MEMORY (nchunks_in_C+2, sizeof (GB_Cp_TYPE), &W_7_mem) ;
        if (W_6 == NULL || W_7 == NULL)
        {
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        JDelta    = ((uint16_t   *) W_6) + 1 ;
        JDeltaSum = ((GB_Cp_TYPE *) W_7) + 1 ;

        // KERNEL LAUNCH 4: phase6
        GB_cuda_ewise_vssp_phase6 <<<grid, block6, 0, stream>>>
            (JDelta, JDeltaSum, Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ()) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // phase 7: construct Cp and Ch  (skipped if C is a vector)
    //--------------------------------------------------------------------------

    int64_t cnvec = 0 ;
    if (C->vdim == 1)
    {
        cnvec = 1 ;
    }
    else
    {
        // exclusive cumsum of JDeltaSum (CPU)
        for (int64_t chunk = 0 ; chunk < nchunks_in_C ; chunk++)
        {
            int64_t s = JDeltaSum [chunk] ;
            JDeltaSum [chunk] = cnvec ;
            cnvec += s ;
        }
        JDeltaSum [nchunks_in_C] = cnvec ;
    }

    // allocate C->p and C->h
    C->p_mem = mem ;
    C->p = GB_MALLOC_MEMORY (cnvec+1, sizeof (GB_Cp_TYPE), &(C->p_mem)) ;
    if (C->vdim == 1)
    {
        C->h = NULL ;
        C->plen = 1 ;
    }
    else
    {
        C->h_mem = mem ;
        C->h = GB_MALLOC_MEMORY (cnvec, sizeof (GB_Cj_TYPE), &(C->h_mem)) ;
        C->plen = cnvec ;
    }
    if (C->p == NULL || (C->h == NULL && C->vdim != 1))
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;

    if (C->vdim == 1)
    {
        GB_Cp_TYPE *Cp = (GB_Cp_TYPE *) C->p ;
        Cp [0] = 0 ;
        Cp [1] = cnz ;
    }
    else
    {
        // KERNEL LAUNCH 5: phase7
        GB_cuda_ewise_vssp_phase7 <<<grid, block6, 0, stream>>>
            (C, JDelta, JDeltaSum, Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ()) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}
