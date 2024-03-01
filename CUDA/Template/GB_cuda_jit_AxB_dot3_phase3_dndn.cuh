//------------------------------------------------------------------------------
// GraphBLAS/CUDA/JitKernels/GB_cuda_jit_AxB_dot3_phase3_dndn.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This CUDA kernel produces the semiring product of two
// dense matrices of types GB_A_TYPE and GB_B_TYPE and common index space size n, to a  
// output matrix of type GB_C_TYPE. The matrices are dense, with uniform
// non-zeros and sparsity patterns. 
// ie. we want to produce C = A'*B in the sense of the given semi-ring.

// This version uses a simple warp-based dense dot product algorithm, when the
// vectors coming from both A and B are dense, for any size of N.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number
// of active threads = min( min(nzA, nzB), 32) 

// Thus, threadblock b owns a semi-ring dot product on a pair of vectors. 
// The work is to load the data, do the multiply and add work and finally 
// reduce this data to a scalar, and write it to Cx[pair].

//------------------------------------------------------------------------------
// warp_ReduceSum_dndn
//------------------------------------------------------------------------------

// FIXME: make this the same static device function
// #include "GB_reduce_whatever.cuh"

__inline__ __device__ GB_Z_TYPE warp_ReduceSum_dndn
(
    thread_block_tile<32> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // FIXME: only works if sizeof(GB_Z_TYPE) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        GB_Z_TYPE next = g.shfl_down( val, i) ;
        GB_ADD( val, val, next ); 
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_dndn_kernel
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase3_dndn_kernel
(
    GrB_Matrix C,   // result matrix
    GrB_Matrix M,   // mask matrix
    GrB_Matrix A,   // input matrix A
    GrB_Matrix B    // input matrix B
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *)A->x  ;
    #endif
    #if !GB_A_IS_PATTERN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *)B->x  ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *)C->x  ;
          int64_t *__restrict__ Ci = C->i ;
    const int64_t *__restrict__ Mi = M->i ;
    #if GB_M_IS_HYPER
    const int64_t *__restrict__ Mh = M->h ;
    #endif
    // A and B are either bitmap or full
    #if GB_A_IS_BITMAP
    const int8_t  *__restrict__ Ab = A->b ;
    #endif
    #if GB_B_IS_BITMAP
    const int8_t  *__restrict__ Bb = B->b ;
    #endif

    // zombie count
    uint64_t zc = 0;

    GB_M_NVALS (mnz) ;

    // total items to be inspected
    int64_t vlen = A->vlen ;
    ASSERT (vlen == B->vlen) ;

    // Main loop over pairs 
    for (int64_t pair_id  = blockIdx.x ; // warp per pair 
                 pair_id  < mnz ;
                 pair_id += gridDim.x)
    {

        // get M(i,j) and C(i,j)
        int64_t i = Mi[pair_id];
        int64_t kk = Ci[pair_id] >> 4;      // FIXME: can remove ">> 4"
        bool cij_exists = false ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity

        // FIXME: test for kk >= 0 not needed if GB_MASK_STRUCT is defined and
        // vlen > 0

        // skip if C(i,j) is a prezombie
        if (kk >= 0)
        {

            // j = kk or j = Mh [kk] if C and M are hypersparse
            int64_t j = GBH_M (Mh, kk) ;

            int64_t pA     = vlen * i ;
            // int64_t pA_end = pA +(A->vlen);

            int64_t pB     = vlen * j ;
            // int64_t pB_end = pB +(B->vlen);

            // convert global data pointer to the local pointer of this block
            GB_DECLAREA (aki) ;
            GB_DECLAREB (bkj) ;

            #if GB_A_IS_FULL && GB_B_IS_FULL
            {
                // FIXME: when both A and B are full, use another method
                // (single pass)
                cij_exists = true ;
                for (int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    GB_MULTADD ( cij, aki, bkj, i, k, j ) ; // cij += aki * bkj
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    int8_t b = (Ab [pA+k] && Bb [pB+k]) ;
                    cij_exists |= b ;
                    if (b)
                    {
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                    }
                }
            }
            #elif GB_A_IS_FULL && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    if (Bb [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                        cij_exists = true ;
                    }
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_FULL
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    if (Ab [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                        cij_exists = true ;
                    }
                }
            }
            #endif
        }

        //----------------------------------------------------------------------
        // reduce per-thread sums to a single scalar
        //----------------------------------------------------------------------

        // Do vote here for control.
        thread_block_tile<32> tile = tiled_partition<32>( this_thread_block() );
        cij_exists = tile.any( cij_exists);
        tile.sync();

        #if !GB_C_ISO
        // FIXME: the ANY monoid needs the cij_exists for each thread
        cij = warp_ReduceSum_dndn<32> ( tile, cij);
        #endif

        // FIXME: if A and B are full, and GB_MASK_STRUCT is true, cij_exists
        // is always true, unless vlen is zero (and then all entries are
        // zombies and there's nothing to do).

        // write result for this block to global mem
        if (threadIdx.x == 0)
        {
            if (cij_exists)
            {
                // Cx [pair_id] = (GB_C_TYPE) cij
                GB_PUTC (cij, Cx, pair_id) ;
                Ci [pair_id] = i ;
            }
            else
            {
                // cij is a zombie
                zc++;
                Ci [pair_id] = GB_FLIP (i) ;
            }
        }
        //__syncthreads ( ) ;

        if( threadIdx.x ==0 && zc > 0)
        {
            GB_cuda_atomic_add <uint64_t>( &(C->nzombies), zc) ;
        }
    }
}

