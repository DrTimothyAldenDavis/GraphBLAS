//------------------------------------------------------------------------------
// GB_AxB_saxpy5_unroll.c: C(I:I+N-1,j)+=A(I:I+N-1,:)*B(:,j)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A fully unrolled kernel for multiplying a bitmap/full submatrix A(I:I+N-1,:)
// times a sparse vector B(:,j) and accumulating the results in a dense column
// subvector C.

// C(I:I+N-1,j) is a single column subvector, as-if-full, of length exactly N
// where N is a small constant.  It consist of rows I, I+1, ... I+N-1 of the
// entire vector C(:,j).

// A(I:I+N-1,:) is a submatrix, held by column, either bitmap or full, of size
// N-by-n, where n can be large.  The submatrix is a subset of rows, I, I+1,
// ... I+N-1, where N is a small constant.

// B is a a single sparse vector from a sparse or hypersparse matrix of length
// n.

//------------------------------------------------------------------------------

    // C (I:I+N-1,j) += A (I:I+N-1,m-1,:)*B(:,j)
    // where N is a small constant.
    {
        GB_CTYPE cx [N] ;
        // cx [0:N-1] = C(I:I+N-1,j)

        #pragma unroll(8)
        for (int t = 0 ; t < N ; t++)
        {
            cx [t] = Cx [pC + I + t] ;
        }

        // cx += A(I:I+N-1,:)*B(:,j)
        // GB_PRAGMA_SIMD
        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
        {
            // bkj = B(k,j)
            const int64_t k = Bi [pB] ;
            GB_GETB (bkj, Bx, pB, B_iso) ;
            // get A(I,k)
            const int64_t pA = (k * m) + I ;
            // cx += A(I:I+N-1,k)*B(k,j)

            GB_MULTADD (cx [0], GBX (Ax, 0 + pA, false), bkj, I+0, k, j) ;
            #if (N > 1)
            GB_MULTADD (cx [1], GBX (Ax, 1 + pA, false), bkj, I+1, k, j) ;
            #endif
            #if (N > 2)
            GB_MULTADD (cx [2], GBX (Ax, 2 + pA, false), bkj, I+2, k, j) ;
            #endif
            #if (N > 3)
            GB_MULTADD (cx [3], GBX (Ax, 3 + pA, false), bkj, I+3, k, j) ;
            #endif
            #if (N > 4)
            GB_MULTADD (cx [4], GBX (Ax, 4 + pA, false), bkj, I+4, k, j) ;
            #endif
            #if (N > 5)
            GB_MULTADD (cx [5], GBX (Ax, 5 + pA, false), bkj, I+5, k, j) ;
            #endif
            #if (N > 6)
            GB_MULTADD (cx [6], GBX (Ax, 6 + pA, false), bkj, I+6, k, j) ;
            #endif
            #if (N > 7)
            GB_MULTADD (cx [7], GBX (Ax, 7 + pA, false), bkj, I+7, k, j) ;
            #endif

            #if 0
            #pragma unroll(8)
            for (int t = 0 ; t < N ; t++)
            {
                // cx [t] += A(I+t,k) * B(k,j)
//              GB_CIJ_MULTADD (t, I + t) ;
                GB_MULTADD (cx [t], GBX (Ax, t + pA, false), bkj, I+t, k, j) ;
            }
            #endif

        }
        // C(I:I+N-1,j) = cx [0:N-1]
        #pragma unroll(8)
        for (int t = 0 ; t < N ; t++)
        {
            Cx [pC + I + t] = cx [t] ;
        }
    }

    #undef N
    #undef I

