//------------------------------------------------------------------------------
// spGEMM_very_sparse_sparse.cu 
//------------------------------------------------------------------------------

// The spGEM_vssp CUDA kernel produces the semi-ring product of two
// sparse matrices of types T_A and T_B and common index space size n, to a  
// output matrix of type T_C. The matrices are sparse, with different numbers
// of non-zeros and different sparsity patterns. 
// ie. we want to produce C = A'*B in the sense of the given semi-ring.

// This version uses a binary-search algorithm, when the sizes nnzA and nnzB
// are far apart in size, neither is very spare nor dense, for any size of N.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number
// of active threads = min( min(nzA, nzB), 32) 

// Thus, each t in threadblock b owns a part of the set of pairs in the 
// sparse-sparse bucket of work. The job for each pair of vectors is to find 
// the intersection of the index sets Ai and Bi, perform the semi-ring dot 
// product on those items in the intersection, and finally
// on exit write it to Cx [pair].

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  GrB_Matrix C         <- result matrix 
//  GrB_Matrix M         <- mask matrix
//  GrB_Matrix A         <- input matrix A
//  GrB_Matrix B         <- input matrix B
#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <cooperative_groups.h>
#include "GB_cuda_kernel.h"

// Using tile size fixed at compile time, we don't need shared memory
#define tile_sz 32 

using namespace cooperative_groups;

template< typename T, int warpSize >
__device__ T reduce_sum(thread_block_tile<warpSize> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    /*
    for (int i = warpSize >> 1; i > 0; i >>= 1)
    {
        val += g.shfl_down(val,i) ;
    }
    */
        val += g.shfl_down(val,16) ;
        val += g.shfl_down(val,8) ;
        val += g.shfl_down(val,4) ;
        val += g.shfl_down(val,2) ;
        val += g.shfl_down(val,1) ;
    return val; // note: only thread 0 will return full sum
}

#define intersects_per_thread 8

template<
    typename T_C, typename T_A, typename T_B,
    typename T_Z, typename T_X, typename T_Y,
    uint64_t srcode>
__global__ void AxB_dot3_phase3_vssp
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,    // do the work defined by Bucket [start:end-1]
    GrB_Matrix C,
    GrB_Matrix M,
    GrB_Matrix A,
    GrB_Matrix B,
    int sz
)
{
    // TODO: Figure out how to use graphblas-specific INFINITY macro
    #ifndef INFINITY
    #define INFINITY std::numeric_limits<T_C>::max()
    #endif

    // Typed pointers to access data in A,B,C
   const T_A *__restrict__ Ax = (T_A*)A->x;
   const T_B *__restrict__ Bx = (T_B*)B->x;
         T_C *__restrict__ Cx = (T_C*)C->x;
         int64_t *__restrict__ Ci = C->i;
   const int64_t *__restrict__ Mi = M->i;
   const int64_t *__restrict__ Ai = A->i;
   const int64_t *__restrict__ Bi = B->i;
   const int64_t *__restrict__ Ap = A->p;
   const int64_t *__restrict__ Bp = B->p;

   // sz = expected non-zeros per dot 
  /* 
   int m = 256/sz;
   int nvecs = end - start;
   int dpt = nvecs/(gridDim.x);
   
   int dots = (nvecs +dpt -1)/dpt; 
   */
   

   // zombie count
   int64_t zc = 0;
   int64_t pair_id;// im;

   // set thread ID
   unsigned int tid_global = threadIdx.x+ blockDim.x* blockIdx.x;

   unsigned long int b = blockIdx.x ;

    // Main loop over pairs in Bucket [start:end-1]
  //for (int64_t kk = start+ tid_global, im = 0; 
  //             kk < end && im < m;  
  //             kk += gridDim.x*blockDim.x, ++im)
    for (int64_t kk = start+ tid_global; 
                 kk < end ;  
                 kk += gridDim.x*blockDim.x)
    {

        pair_id = Bucket[ kk ];

        int64_t i = Mi[pair_id];
        int64_t j = Ci[pair_id] >> 4;

        int64_t pA      = Ap[i];
        int64_t pA_end  = Ap[i+1];
        int64_t nnzA = pA_end - pA;

        int64_t pB      = Bp[j];
        int64_t pB_end  = Bp[j+1];
        int64_t nnzB = pB_end - pB;

        //Search for each nonzero in the smaller vector to find intersection 
        bool cij_exists = false;

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        T_Z cij = GB_IDENTITY ;

        if (nnzA <= nnzB)
        {
            //------------------------------------------------------------------
            // A(:,i) is very sparse compared to B(:,j)
            //------------------------------------------------------------------

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                 /*
                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    pA++ ;
                }
                */
                pA += ( ia < ib );
                if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    // discard all entries B(ib:ia-1,j)
                    int64_t pleft = pB + 1 ;
                    int64_t pright = pB_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                    //ASSERT (pleft > pB) ;
                    pB = pleft ;
                }
                else if (ia == ib) // ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT_MERGE (pA, pB);
                    //GB_DOT_TERMINAL (cij) ;   // break if cij == terminal
                    pA++ ;
                    pB++ ;
                }
            }
        }
        else
        {
            //------------------------------------------------------------------
            // B(:,j) is very sparse compared to A(:,i)
            //------------------------------------------------------------------

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;

                pB += ( ib < ia);

                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    // discard all entries A(ia:ib-1,i)
                    int64_t pleft = pA + 1 ;
                    int64_t pright = pA_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                    //ASSERT (pleft > pA) ;
                    pA = pleft ;
                }
                /*
                else if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    pB++ ;
                }
                */
                else if (ia == ib)// ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT_MERGE (pA, pB) ;
                    //GB_DOT_TERMINAL (cij) ;   // break if cij == terminal
                    pA++ ;
                    pB++ ;
                }
            }

        }
        GB_CIJ_EXIST_POSTCHECK ;
        if ( cij_exists){
           Ci[pair_id] = i ;
           GB_PUTC ( Cx[pair_id] = (T_C)cij ) ;
        }
        else {
           zc++; 
           //printf(" %lld, %lld is zombie %d!\n",i,j,zc);
           Ci[pair_id] = GB_FLIP( i ) ;
        }


    }
    this_thread_block().sync();

    //--------------------------------------------------------------------------
    // reduce sum per-thread values to a single scalar
    //--------------------------------------------------------------------------
    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());
    zc = reduce_sum<int,tile_sz>(tile, zc);

    if( threadIdx.x ==0 && zc > 0) {
      //printf("vssp warp %d zombie count = %d\n", blockIdx.x, zc);
      atomicAdd( (unsigned long long int*)&(C->nzombies), (unsigned long long int)zc);
      //printf(" vssp Czombie = %lld\n",C->nzombies);
    }

}

