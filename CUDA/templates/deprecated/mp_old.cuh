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
T warp_Reduce_Op(thread_block_tile<warp_sz> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // Temporary T is necessary to handle arbirary ops
    /*
    for (int i = warp_sz >> 1; i > 0; i >>= 1)
    {
        T next = g.shfl_down( val, i);
        val = GB_ADD( val, next ) ;
    }
    */
        T next = g.shfl_down( val, 16);
        val = GB_ADD( val, next ) ;
        next = g.shfl_down( val, 8);
        val = GB_ADD( val, next ) ;
        next = g.shfl_down( val, 4);
        val = GB_ADD( val, next ) ;
        next = g.shfl_down( val, 2);
        val = GB_ADD( val, next ) ;
        next = g.shfl_down( val, 1);
        val = GB_ADD( val, next ) ;
    return val;
}

template<typename T, int warpSize>
__inline__ __device__
T block_Reduce_Op(thread_block g, T val)
{
  static __shared__ T shared[warpSize]; // Shared mem for 32 partial sums
  

  int lane = threadIdx.x & 31 ; // % warpSize;
  int wid  = threadIdx.x >> 5 ; // / warpSize;
  thread_block_tile<warpSize> tile = tiled_partition<warpSize>( g );

  // Each warp performs partial reduction
  val = warp_Reduce_Op<T, warpSize>( tile, val);    

  // Wait for all partial reductions
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory
  __syncthreads();              // Wait for all partial reductions

  //if (wid > 0 ) return val;

  //read from shared memory only if that warp existed
  val = (threadIdx.x <  (blockDim.x / warpSize ) ) ? shared[lane] : GB_IDENTITY;

  if (wid==0) val = warp_Reduce_Op<T, warpSize>( tile, val); //Final reduce within first warp

  return val;
}

/*
template< typename T, int warp_sz>
__device__ __inline__ 
T reduce_plus(thread_block_tile<warp_sz> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    //for (int i = warp_sz >> 1; i > 0; i >>= 1)
    //{
    //    val += g.shfl_down( val, i) ;
    //}
        val += g.shfl_down(val,16) ;
        val += g.shfl_down(val,8) ;
        val += g.shfl_down(val,4) ;
        val += g.shfl_down(val,2) ;
        val += g.shfl_down(val,1) ;
    return val; // note: only thread 0 will return full sum and flag value
}
*/


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
    int zc = 0;

    int64_t pair_id;

    // set thread ID
    //int tid_global = threadIdx.x+ blockDim.x* blockIdx.x;
    int tid = threadIdx.x;

    int b = blockIdx.x ;

    // total items to be inspected
    int64_t nnzA = 0;
    int64_t nnzB = 0;

    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());

    // int parts = blockDim.x; // whole block (at least 32 threads) per dot product
// int has_zombies = 0 ;

    // Main loop over pairs 
    int64_t kk ;
    for (kk = start+ blockIdx.x; //block per pair 
         kk < end;  
         kk += gridDim.x )
    {

         pair_id = Bucket [kk] ;
         int64_t i = Mi[pair_id];
         int64_t j = Ci[pair_id] >> 4;

         // find A(:,i)
         int64_t xstart = Ap[i];        // pA_start
         int64_t xend   = Ap[i+1];      // pA_end
         nnzA = xend - xstart;          // ainz

         // find B(:,j)
         int64_t ystart = Bp[j];        // pB_start
         int64_t yend   = Bp[j+1];      // pB_ed
         nnzB = yend - ystart;          // bjnz

//       if (threadIdx.x == 0 )
//       {
//          printf ("\nblockId: %d Computing (%ld,%ld)\n",blockIdx.x, i, j) ;
//          printf ("\nA(:,%ld): nnzA %ld\n", i, nnzA) ;
//          for (int64_t p = xstart ; p < xend ; p++) printf ("  %ld: %ld\n", p, Ai [p]) ;
//          printf ("\nB(:,%ld): nnzB %ld\n", j, nnzB) ;
//          for (int64_t p = ystart ; p < yend ; p++) printf ("  %ld: %ld\n", p, Bi [p]) ;
//       }
//       this_thread_block().sync();

//         if(threadIdx.x == 0 && j == 139 && i == 945)
//             printf("blk%d tid=%d, nnzA=%d, nnzB=%d\n", blockIdx.x, tid_global, nnzA, nnzB);
//
    /* 
    if (threadIdx.x ==0 ) {
      printf("block %d  doing dot %lld  i,j= %lld,%lld\n", blockIdx.x, pair_id, i, j);
    }
    */
    //we want more than one intersection per thread
    int64_t nxy = nnzA + nnzB;

    int work_per_thread = (nxy +blockDim.x -1)/blockDim.x;
    int diag = GB_IMIN( work_per_thread*tid, nxy);
    int diag_end = GB_IMIN( diag + work_per_thread, nxy);
    //printf(" thd%d parts = %u wpt = %u diag, diag_end  = %u,%u\n",tid, parts, work_per_thread, diag, diag_end); 

//  if (mydump && threadIdx.x == 0)
//  {
//      printf ("work_per_thread %d nxy %ld parts %d diag %d diag_end %d\n",
//          work_per_thread, nxy, parts, diag, diag_end) ;
//  }
    
    int x_min = GB_IMAX( (diag - nnzB), 0);
    int x_max = GB_IMIN( diag, nnzA);

    //printf("start thd%u x_min = %u x_max = %u\n", tid_global, x_min,x_max);
    while ( x_min < x_max) { //binary search for correct diag break
      int pivot = (x_min +x_max)/2;
      if ( Ai[pivot + xstart] < Bi[ diag -pivot -1 + ystart]) {
         x_min = pivot +1;
      }
      else {
         x_max = pivot;
      }
    }
    int xcoord = x_min;
    int ycoord = diag -x_min -1;
    if ( ( diag > 0) 
      && (diag < nxy ) 
      && (ycoord >= 0 ) 
      && (Ai[xcoord+xstart] == Bi[ycoord+ystart]) 
      ) 
    { 
       diag--; //adjust for intersection incrementing both pointers 
    }
    // two start points are known now
    int tx_start = xcoord +xstart;
    int ty_start = diag -xcoord +ystart; 

    //if (x_start != y_start)
    //   printf("start thd%u  xs,ys = %i,%i\n", tid_global, x_start, y_start);

    x_min = GB_IMAX( (int)(diag_end - nnzB), 0);
    x_max = GB_IMIN( diag_end, nnzA);

    while ( x_min < x_max) {
       int pivot = (x_min +x_max)/2;
       //printf("thd%u pre_sw piv=%u diag_e = %u  xmin,xmax=%u,%u\n", tid_global, pivot, diag_end,x_min, x_max);
       if ( Ai[pivot+ xstart] < Bi[ diag_end -pivot -1 +ystart]) {
          x_min = pivot +1;
       }
       else {
          x_max = pivot;
       }
       //printf("thd%u piv=%u xmin,xmax = %u,%u\n", tid_global, pivot, x_min, x_max);
    }
    xcoord = x_min;
    ycoord = diag_end -x_min -1;
    if ( (diag_end < nxy) 
      && (ycoord > 0)
      && (Ai[xcoord +xstart] == Bi[ycoord + ystart]) 
      ) { 
        diag--; //adjust for intersection incrementing both pointers  
    }
    // two end points are known now
    int tx_end = xcoord +xstart; 
    int ty_end = diag_end - xcoord + ystart; 

    GB_DECLAREA (aki) ;
    GB_DECLAREB (bkj) ;
    T_Z cij = GB_IDENTITY ;

    // TODO PLUS_PAIR_INT64, FP32, FP64: no need for cij_exists.
    // just check if cij > 0

    int cij_exists  = 0 ;
    //printf(" thd%u has init value %f\n",tid, cij);

    //merge-path dot product
    int64_t pA = tx_start;       // pA
    int64_t pB = ty_start;       // pB

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

    while ( pA < tx_end && pB < ty_end ) // && nnzA != 0 && nnzB != 0)
    {
        if (Ai [pA] == Bi [pB])
        {
            GB_GETA (aki, Ax, k) ;      // aki = Ax [k]
            GB_GETB (bkj, Bx, l) ;      // bkj = Bx [l]
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
       cij = warp_Reduce_Op<T_Z, tile_sz>( tile, cij );
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
  }
  //this_thread_block().sync();  //needed for multi-warp correctness

//--------------------------------------------------------------------------

  if( tid ==0 && zc > 0)
  {
//      printf("warp %d zombie count = %d, nzombies = %d\n", blockIdx.x, zc, C->nzombies);
      atomicAdd( (unsigned long long int*)&(C->nzombies), (unsigned long long int)zc);
//      printf(" Czombie = %lld\n",C->nzombies);
  }

  //__syncthreads();

}

