//------------------------------------------------------------------------------
// AxB_dot3_phase3_mp.cu 
//------------------------------------------------------------------------------

// This CUDA kernel produces the semi-ring product of two
// sparse matrices of types T_A and T_B and common index space size n, to a  
// output matrix of type T_C. The matrices are sparse, with different numbers
// of non-zeros and different sparsity patterns. 
// ie. we want to produce C = A'*B in the sense of the given semi-ring.

// This version uses a merge-path algorithm, when the sizes nnzA and nnzB are 
// relatively close in size, neither is very sparse nor dense, for any size of N.
// Handles arbitrary sparsity patterns with guaranteed load balance.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number
// of active threads = min( min(g_xnz, g_ynz), 32) 

// Thus, threadblock b owns a part of the index set spanned by g_xi and g_yi.  Its job
// is to find the intersection of the index sets g_xi and g_yi, perform the semi-ring dot
// product on those items in the intersection, and finally reduce this data to a scalar, 
// on exit write it to g_odata [b].

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  matrix<T_C> *C         <- result matrix 
//  matrix<T_M> *M         <- mask matrix
//  matrix<T_A> *A         <- input matrix A
//  matrix<T_B> *B         <- input matrix B

#pragma once

#include <limits>
#include <cstdint>
#include <cooperative_groups.h>
#include "GB_cuda_kernel.h"

// Using tile size fixed at compile time, we don't need shared memory
#define tile_sz 32 

using namespace cooperative_groups;

template< typename T, int warp_sz>
__device__ __inline__ 
T GB_reduce_sum(thread_block_tile<warp_sz> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // Temporary T is necessary to handle arbirary ops
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        T next = g.shfl_down( val, i);
        val = GB_ADD( val, next ) ;
    }
    return val;
}

template< typename T, int warp_sz>
__device__ __inline__ 
T reduce_plus(thread_block_tile<warp_sz> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down( val, i) ;
    }
    return val; // note: only thread 0 will return full sum and flag value
}


template<
    typename T_C, typename T_A, typename T_B,
    typename T_Z, typename T_X, typename T_Y,
    uint64_t srcode>
__global__ void AxB_dot3_phase3_mp
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,    // do the work in Bucket [start:end-1]
    GrB_Matrix C,
    GrB_Matrix M,
    GrB_Matrix A,
    GrB_Matrix B,
    int sz
)
{

    C->jumbled = true;
    const T_A *__restrict__ Ax = (T_A *)A->x  ;
    const T_B *__restrict__ Bx = (T_B *)B->x  ;
          T_C *__restrict__ Cx = (T_C *)C->x  ;
      int64_t *__restrict__ Ci = C->i ;
    const int64_t *__restrict__ Mi = M->i ;
    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Bi = B->i ;
    const int64_t *__restrict__ Ap = A->p ;
    const int64_t *__restrict__ Bp = B->p ;

    // zombie count
    int64_t zc = 0;

    int64_t pair_id;

    // set thread ID
    int tid_global = threadIdx.x+ blockDim.x* blockIdx.x;
    int tid = threadIdx.x;

    int b = blockIdx.x ;

    // total items to be inspected
    int64_t ainz = 0;
    int64_t bjnz = 0;
    int64_t n_intersect = 0;

    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());

    int parts = blockDim.x; // whole block (at least 32 threads) per dot product

    // int has_zombies = 0 ;

    // Main loop over pairs 
    int64_t kk ;
    for (kk = start+ blockIdx.x; //warp per pair 
         kk < end;  
         kk += gridDim.x )
    {

         pair_id = Bucket [kk] ;
         int64_t i = Mi[pair_id];
         int64_t j = Ci[pair_id] >> 4;


         // find A(:,i)
         int64_t pA_start = Ap[i];        // pA_start
         int64_t pA_end   = Ap[i+1];      // pA_end
         ainz = pA_end - pA_start;          // ainz
         
         #define shared_vector_size 512
         __shared__ int64_t Ai_s[shared_vector_size];
         bool use_A_shared = false;

         if (ainz < shared_vector_size)
         {
            use_A_shared = true;
            #pragma unroll
            for ( int i = pA_start +tid; i< pA_end; i+= blockDim.x)
            {   Ai_s[i- pA_start] = Ai[ i];
            }   
            __syncthreads();
         }
         

         // find B(:,j)
         int64_t pB_start = Bp[j];        // pB_start
         int64_t pB_end   = Bp[j+1];      // pB_end
         bjnz = pB_end - pB_start;          // bjnz
         
         __shared__ int64_t Bj_s[shared_vector_size];
         bool use_B_shared = false;

         if (bjnz < shared_vector_size)
         {
            use_B_shared = true;
            #pragma unroll
            for ( int i = pB_start +tid; i< pB_end; i+= blockDim.x)
            {   Bj_s[i- pB_start] = Bi[ i];
            }   
            __syncthreads();
         }
        

//       if (threadIdx.x == 0 && mydump)
//       {
//          printf ("\nComputing (%ld,%ld)\n", i, j) ;
//          printf ("\nA(:,%ld): nnzA %ld\n", i, nnzA) ;
//          for (int64_t p = xstart ; p < xend ; p++) printf ("  %ld: %ld\n", p, Ai [p]) ;
//          printf ("\nB(:,%ld): nnzB %ld\n", j, nnzB) ;
//          for (int64_t p = ystart ; p < yend ; p++) printf ("  %ld: %ld\n", p, Bi [p]) ;
//       }

//         if(threadIdx.x == 0 && j == 139 && i == 945)
//             printf("blk%d tid=%d, nnzA=%d, nnzB=%d\n", blockIdx.x, tid_global, nnzA, nnzB);
//
         n_intersect = GB_IMIN( pA_end -pA_start, pB_end -pB_start); 
    /* 
    if (threadIdx.x ==0 ) {
      printf("block %d  doing dot %lld  i,j= %lld,%lld\n", blockIdx.x, pair_id, i, j);
    }
    */
    //we want more than one intersection per thread
    int64_t nxy = ainz + bjnz;

    int work_per_thread = (nxy +parts -1)/parts;
    int diag = GB_IMIN( work_per_thread*tid, nxy);
    int diag_end = GB_IMIN( diag + work_per_thread, nxy);
    //printf(" thd%d parts = %u wpt = %u diag, diag_end  = %u,%u\n",tid, parts, work_per_thread, diag, diag_end); 

//  if (mydump && threadIdx.x == 0)
//  {
//      printf ("work_per_thread %d nxy %ld parts %d diag %d diag_end %d\n",
//          work_per_thread, nxy, parts, diag, diag_end) ;
//  }
    
    int x_min = GB_IMAX( diag - bjnz , 0);
    int x_max = GB_IMIN( diag, ainz);

    //printf("start thd%u x_min = %u x_max = %u\n", tid_global, x_min,x_max);
    while ( x_min < x_max) { //binary search for correct diag break
      int pivot = (x_min +x_max) >> 1;
      if ( use_A_shared && use_B_shared ) 
      {
          if ( Ai_s[pivot ] < Bj_s[ diag -pivot -1 ]) {
             x_min = pivot +1;
          }
          else {
             x_max = pivot;
          }

      }
      else {
          if ( Ai[pivot + pA_start] < Bi[ diag -pivot -1 + pB_start]) {
             x_min = pivot +1;
          }
          else {
             x_max = pivot;
          }
      }
      
       /*
          if ( Ai[pivot + pA_start] < Bi[ diag -pivot -1 + pB_start]) {
             x_min = pivot +1;
          }
          else {
             x_max = pivot;
          }
       */
    }
    int xcoord = x_min;
    int ycoord = diag -x_min -1;
    int64_t Atest;
    int64_t Btest;
    if ( use_A_shared && use_B_shared)
    {
       Atest = Ai_s[xcoord];
       Btest = Bj_s[ycoord];
    }
    else
    {
       Atest = Ai[xcoord+pA_start];
       Btest = Bi[ycoord+pB_start];
    }
    if ( ( diag > 0) 
      && (diag < nxy ) 
      && (Atest == Btest)
      && (ycoord >= 0 ) 
      ) 
    { 
       diag--; //adjust for intersection incrementing both pointers 
    }
    // two start points are known now
    int tx_start = xcoord +pA_start;
    int ty_start = diag -xcoord +pB_start; 

    //if (x_start != y_start)
    //   printf("start thd%u  xs,ys = %i,%i\n", tid_global, x_start, y_start);

    x_min = GB_IMAX( diag_end - bjnz, 0);
    x_max = GB_IMIN( diag_end, ainz);

    while ( x_min < x_max) {
      int pivot = (x_min +x_max) >> 1;
      
      if ( use_A_shared && use_B_shared ) 
      {
          if ( Ai_s[pivot ] < Bj_s[ diag_end -pivot -1 ]) {
             x_min = pivot +1;
          }
          else {
             x_max = pivot;
          }

      }
      else {
          if ( Ai[pivot + pA_start] < Bi[ diag_end -pivot -1 + pB_start]) {
             x_min = pivot +1;
          }
          else {
             x_max = pivot;
          }
      }
      /*
       //printf("thd%u pre_sw piv=%u diag_e = %u  xmin,xmax=%u,%u\n", tid_global, pivot, diag_end,x_min, x_max);
       if ( Ai[pivot+ pA_start] < Bi[ diag_end -pivot -1 +pB_start]) {
          x_min = pivot +1;
       }
       else {
          x_max = pivot;
       }
       */
       
       //printf("thd%u piv=%u xmin,xmax = %u,%u\n", tid_global, pivot, x_min, x_max);
    }
    xcoord = x_min;
    ycoord = diag_end -x_min -1;
    if ( use_A_shared && use_B_shared)
    {
       Atest = Ai_s[xcoord];
       Btest = Bj_s[ycoord];
    }
    else
    {
       Atest = Ai[xcoord+pA_start];
       Btest = Bi[ycoord+pB_start];
    }
    if ( (diag_end > 0)
      && (diag_end < nxy) 
      && (ycoord >= 0)
      && (Atest == Btest)
      ) 
      { 
        diag--; //adjust for intersection incrementing both pointers  
      }
    // two end points are known now
    int tx_end = xcoord +pA_start; 
    int ty_end = diag_end - xcoord + pB_start; 

    GB_DECLAREA (aki) ;
    GB_DECLAREB (bkj) ;
    T_Z cij = GB_IDENTITY ;

    // TODO PLUS_PAIR_INT64, FP32, FP64: no need for cij_exists.
    // just check if cij > 0

    int cij_exists  = 0 ;
    //printf(" thd%u has init value %f\n",tid, cij);

    //merge-path dot product
    int pA = tx_start;       // pA
    int pB = ty_start;       // pB

    if (use_A_shared && use_B_shared)
    {
       pA -=  pA_start;
       pB -=  pB_start;
       tx_end -= pA_start;
       ty_end -= pB_start;

    }
//  if (mydump) //  && threadIdx.x == 0)
//  {
//      printf ("%d tx_start %d\n", threadIdx.x, tx_start) ;
//      printf ("%d tx_end   %d\n", threadIdx.x, tx_end  ) ;
//      printf ("%d ty_start %d\n", threadIdx.x, ty_start) ;
//      printf ("%d ty_end   %d\n", threadIdx.x, ty_end  ) ;
//  }

//    if(threadIdx.x == 0 && j == 139) {
//        printf("blk%d, thd%d k=%d, l=%d, tx_start=%d, ty_start=%d, tx_end=%d, ty_end=%d\n", blockIdx.x, tid_global, k, l, tx_start, ty_start, tx_end, ty_end);
//    }

    while ( pA < tx_end && pB < ty_end ) 
    {
      if( use_A_shared && use_B_shared)
      {
        if (Ai_s [pA] == Bj_s[pB])
        {
            GB_GETA (aki, Ax, pA) ;      // aki = Ax [k]
            GB_GETB (bkj, Bx, pB) ;      // bkj = Bx [l]
            // if (cij_exists)
            {
                // HACK: cij_exists = 1 ;
                GB_MULTADD (cij, aki, bkj) ;   // cij += aki * bkj
            }
            // TODO check terminal condition
            pA++ ;
            pB++ ;
//                if(j == 139 && i == 945)
//                    printf(" block%u work value = %d, exists = %d\n", b, cij, cij_exists);
        }
        else
        {
            pA += ( Ai_s[pA] < Bj_s[pB] ) ;
            pB += ( Ai_s[pA] > Bj_s[pB] ) ;
        }


      }
      else {
        if (Ai [pA] == Bi [pB])
        {
            GB_GETA (aki, Ax, pA) ;      // aki = Ax [k]
            GB_GETB (bkj, Bx, pB) ;      // bkj = Bx [l]
            // if (cij_exists)
            {
                // HACK: cij_exists = 1 ;
                GB_MULTADD (cij, aki, bkj) ;   // cij += aki * bkj
//                    if(j == 139 && i == 945)
//                        printf("blk%d thd%d ix at %lld  %lld cij += %d * %d \n", blockIdx.x, tid_global, Ai[k], Bi[l], aki, bkj);
            }
#if 0
            else
            {
                cij_exists = 1 ;
//              if (mydump) printf ("%d Found k: %d, l %d, Ai[k]: %ld, Bi[l]:%ld\n",
//                  threadIdx.x, k, l, Ai [k], Bi [l]) ;
                GB_C_MULT (cij, aki, bkj) ;    // cij = aki * bkj
//                    if(j == 139 && i == 945)
//                        printf("blk%d thd%d ix at %lld %lld  cij = %d * %d, k=%d, l=%d i=%lld j=%lld \n", blockIdx.x, tid_global, Ai[k], Bi[l], Ax[k], Bx[l], k, l, i, j);
            }
#endif
            // TODO check terminal condition
            pA++ ;
            pB++ ;
//                if(j == 139 && i == 945)
//                    printf(" block%u work value = %d, exists = %d\n", b, cij, cij_exists);
        }
        else
        {
            pA += ( Ai[pA] < Bi[pB] ) ;
            pB += ( Ai[pA] > Bi[pB] ) ;
        }
      }
    }

    //tile.sync( ) ;
    //--------------------------------------------------------------------------
    // reduce sum per-thread values to a single scalar, get OR of flag
    //--------------------------------------------------------------------------
    /*
    if (tid == 0)
    {
        printf ("reduce %d : %d exists = %d\n", b,  cij, cij_exists) ;
    }
    __syncthreads();
    */

    // Do vote here for control.
    // HACK for PLUS_PAIR:
    cij_exists = (cij > 0) ;
    cij_exists = tile.any( cij_exists);
    //tile.sync();

    #if !GB_C_ISO
    if (cij_exists)
    {
       cij = GB_reduce_sum<T_Z, tile_sz>( tile, cij );
    }
    #endif
    // else has_zombies = 1;


        //__syncthreads();
    //tile.sync( );
    // write result for this block to global mem
    if (tid == 0)
    {
        //printf ("final %d : %d exists = %d\n", b,  cij, cij_exists) ;
//      if (mydump) printf ("Result for (%ld,%ld): %d\n", i, j, cij_exists); 

        if (cij_exists)
        {
//
//            if(j == 139) {
//                printf("what's the deal here? %d, %ld\n", cij, i);
//            }

            //printf(" cij = %d\n", cij);
           GB_PUTC ( Cx[pair_id]=(T_C)cij ) ;
           Ci[pair_id] = i ;
        }
        else
        {
           // printf(" dot %d is a zombie\n", pair_id);
           zc++;
           Ci[pair_id]=GB_FLIP (i) ;
        }
    }
    //__syncthreads(); 
  }

//--------------------------------------------------------------------------

  if( tid ==0 && zc > 0)
  {
//      printf("warp %d zombie count = %d, nzombies = %d\n", blockIdx.x, zc, C->nzombies);
      atomicAdd( (unsigned long long int*)&(C->nzombies), (unsigned long long int)zc);
//      printf(" Czombie = %lld\n",C->nzombies);
  }

  //__syncthreads();

}

