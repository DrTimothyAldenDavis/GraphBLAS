//------------------------------------------------------------------------------
// GB_AxB_saxpy5_template.c: C+=A*B when C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is as-if-full.
// A is bitmap or full.
// B is sparse or hypersparse.

// GB_CIJ_MULTADD:  cx [s] += A(i,k) * B(k,j)
#if A_IS_BITMAP
    #define GB_CIJ_MULTADD(s,i)                                       \
    if (Ab [s + pA])                                                  \
    {                                                                 \
        GB_MULTADD (cx [s], GBX (Ax, s + pA, A_iso), bkj, i, k, j) ;  \
    }
#else
    #define GB_CIJ_MULTADD(s,i)                                       \
        GB_MULTADD (cx [s], GBX (Ax, s + pA, A_iso), bkj, i, k, j) ;
#endif

{

#ifdef GB_AVX512F
typedef GB_CTYPE __attribute__ ((vector_size (8 * sizeof (GB_CTYPE)))) v8 ;
typedef GB_CTYPE __attribute__ ((vector_size (4 * sizeof (GB_CTYPE)))) v4 ;
typedef GB_CTYPE __attribute__ ((vector_size (8 * sizeof (GB_CTYPE)), aligned (8))) v8u ;
typedef GB_CTYPE __attribute__ ((vector_size (4 * sizeof (GB_CTYPE)), aligned (8))) v4u ;
#endif


    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        // get the task descriptor
        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start ; jB < jB_end ; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBH (Bh, jB) ;
            GB_CTYPE *restrict Cxj = Cx + (j * m) ;
            const int64_t pB_start = Bp [jB] ;
            const int64_t pB_end   = Bp [jB+1] ;
            // C(:,j) += A*B(:,j), on sets of rows of C and A at a time

            for (int64_t i = 0 ; i < m - 15 ; i += 16)
            {
                // get C(i:i+15,j)
                #ifdef GB_AVX512F
                v8 c1 = (*((v8u *) (Cxj + i    ))) ;
                v8 c2 = (*((v8u *) (Cxj + i + 8))) ;
                #else
                GB_CTYPE cx [16] ;
                memcpy (cx, Cxj + i, 16 * sizeof (GB_CTYPE)) ;
                #endif
                // get A(i,0)
                const GB_ATYPE *restrict Axi = Ax + i ;
                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                {
                    // bkj = B(k,j)
                    const int64_t k = Bi [pB] ;
                    const GB_BTYPE bkj = Bx [pB] ;
                    // get A(i,k)
                    const GB_ATYPE *restrict ax = Axi + (k * m) ;
                    // C(i:i+15,j) += A(i:i+15,k)*B(k,j)
                    #ifdef GB_AVX512F
                    c1 += (*((v8u *) (ax    ))) * bkj ;
                    c2 += (*((v8u *) (ax + 8))) * bkj ;
                    #else
                    cx [ 0] += ax [ 0] * bkj ;
                    cx [ 1] += ax [ 1] * bkj ;
                    cx [ 2] += ax [ 2] * bkj ;
                    cx [ 3] += ax [ 3] * bkj ;
                    cx [ 4] += ax [ 4] * bkj ;
                    cx [ 5] += ax [ 5] * bkj ;
                    cx [ 6] += ax [ 6] * bkj ;
                    cx [ 7] += ax [ 7] * bkj ;
                    cx [ 8] += ax [ 8] * bkj ;
                    cx [ 9] += ax [ 9] * bkj ;
                    cx [10] += ax [10] * bkj ;
                    cx [11] += ax [11] * bkj ;
                    cx [12] += ax [12] * bkj ;
                    cx [13] += ax [13] * bkj ;
                    cx [14] += ax [14] * bkj ;
                    cx [15] += ax [15] * bkj ;
                    #endif
                }
                // save C(i:i+15,j)
                #ifdef GB_AVX512F
                (*((v8u *) (Cxj + i    ))) = c1 ;
                (*((v8u *) (Cxj + i + 8))) = c2 ;
                #else
                memcpy (Cxj + i, cx, 16 * sizeof (GB_CTYPE)) ;
                #endif
            }

            switch (m & 15)
            {

                case 15:
                    {
                        // C(m-15:m-1,j) += A(m-15:m-1,j)*B(:,j)
                        GB_CTYPE cx [15] ;
                        memcpy (cx, Cxj + m-15, 15 * sizeof (GB_CTYPE)) ;
                        // get A(m-15,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 15 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-15,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-15:m-1,j) += A(m-15:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                            cx [10] += ax [10] * bkj ;
                            cx [11] += ax [11] * bkj ;
                            cx [12] += ax [12] * bkj ;
                            cx [13] += ax [13] * bkj ;
                            cx [14] += ax [14] * bkj ;
                        }
                        memcpy (Cxj + m-15, cx, 15 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 14:
                    {
                        // C(m-14:m-1,j) += A(m-14:m-1,j)*B(:,j)
                        GB_CTYPE cx [14] ;
                        memcpy (cx, Cxj + m-14, 14 * sizeof (GB_CTYPE)) ;
                        // get A(m-14,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 14 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-14,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-14:m-1,j) += A(m-14:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                            cx [10] += ax [10] * bkj ;
                            cx [11] += ax [11] * bkj ;
                            cx [12] += ax [12] * bkj ;
                            cx [13] += ax [13] * bkj ;
                        }
                        memcpy (Cxj + m-14, cx, 14 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 13:
                    {
                        // C(m-13:m-1,j) += A(m-13:m-1,j)*B(:,j)
                        GB_CTYPE cx [13] ;
                        memcpy (cx, Cxj + m-13, 13 * sizeof (GB_CTYPE)) ;
                        // get A(m-13,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 13 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-13,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-13:m-1,j) += A(m-13:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                            cx [10] += ax [10] * bkj ;
                            cx [11] += ax [11] * bkj ;
                            cx [12] += ax [12] * bkj ;
                        }
                        memcpy (Cxj + m-13, cx, 13 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 12:
                    {
                        // C(m-12:m-1,j) += A(m-12:m-1,j)*B(:,j)
                        GB_CTYPE cx [12] ;
                        memcpy (cx, Cxj + m-12, 12 * sizeof (GB_CTYPE)) ;
                        // get A(m-12,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 12 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-12,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-12:m-1,j) += A(m-12:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                            cx [10] += ax [10] * bkj ;
                            cx [11] += ax [11] * bkj ;
                        }
                        memcpy (Cxj + m-12, cx, 12 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 11:
                    {
                        // C(m-11:m-1,j) += A(m-11:m-1,j)*B(:,j)
                        GB_CTYPE cx [11] ;
                        memcpy (cx, Cxj + m-11, 11 * sizeof (GB_CTYPE)) ;
                        // get A(m-11,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 11 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-11,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-11:m-1,j) += A(m-11:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                            cx [10] += ax [10] * bkj ;
                        }
                        memcpy (Cxj + m-11, cx, 11 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 10:
                    {
                        // C(m-10:m-1,j) += A(m-10:m-1,j)*B(:,j)
                        GB_CTYPE cx [10] ;
                        memcpy (cx, Cxj + m-10, 10 * sizeof (GB_CTYPE)) ;
                        // get A(m-10,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 10 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-10,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-10:m-1,j) += A(m-10:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                            cx [9] += ax [9] * bkj ;
                        }
                        memcpy (Cxj + m-10, cx, 10 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 9:
                    {
                        // C(m-9:m-1,j) += A(m-9:m-1,j)*B(:,j)
                        GB_CTYPE cx [9] ;
                        memcpy (cx, Cxj + m-9, 9 * sizeof (GB_CTYPE)) ;
                        // get A(m-9,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 9 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-9,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-9:m-1,j) += A(m-9:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            cx [8] += ax [8] * bkj ;
                        }
                        memcpy (Cxj + m-9, cx, 9 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 8:
                    {
                        // C(m-8:m-1,j) += A(m-8:m-1,j)*B(:,j)
                        // get C(m-8:m-1,j)
                        #ifdef GB_AVX512F
                        v8 c1 = (*((v8u *) (Cxj + m-8))) ;
                        #else
                        GB_CTYPE cx [8] ;
                        memcpy (cx, Cxj + m-8, 8 * sizeof (GB_CTYPE)) ;
                        #endif
                        // get A(m-8,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 8 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-8,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-8:m-1,j) += A(m-8:m-1,k)*B(k,j)
                            #ifdef GB_AVX512F
                            c1 += (*((v8u *) ax)) * bkj ;
                            #else
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                            cx [7] += ax [7] * bkj ;
                            #endif
                        }
                        // save C(m-8:m-1,j)
                        #ifdef GB_AVX512F
                        (*((v8u *) (Cxj + m-8))) = c1 ;
                        #else
                        memcpy (Cxj + m-8, cx, 8 * sizeof (GB_CTYPE)) ;
                        #endif
                    }
                    break ;

                case 7:
                    {
                        // C(m-7:m-1,j) += A(m-7:m-1,j)*B(:,j)
                        GB_CTYPE cx [7] ;
                        memcpy (cx, Cxj + m-7, 7 * sizeof (GB_CTYPE)) ;
                        // get A(m-7,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 7 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-7,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-7:m-1,j) += A(m-7:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                            cx [6] += ax [6] * bkj ;
                        }
                        memcpy (Cxj + m-7, cx, 7 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 6:
                    {
                        // C(m-6:m-1,j) += A(m-6:m-1,j)*B(:,j)
                        GB_CTYPE cx [6] ;
                        memcpy (cx, Cxj + m-6, 6 * sizeof (GB_CTYPE)) ;
                        // get A(m-6,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 6 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-6,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-6:m-1,j) += A(m-6:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                            cx [5] += ax [5] * bkj ;
                        }
                        memcpy (Cxj + m-6, cx, 6 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 5:
                    {
                        // C(m-5:m-1,j) += A(m-5:m-1,j)*B(:,j)
                        GB_CTYPE cx [5] ;
                        memcpy (cx, Cxj + m-5, 5 * sizeof (GB_CTYPE)) ;
                        // get A(m-5,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 5 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-5,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-5:m-1,j) += A(m-5:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            cx [4] += ax [4] * bkj ;
                        }
                        memcpy (Cxj + m-5, cx, 5 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 4:
                    {
                        // C(m-4:m-1,j) += A(m-4:m-1,j)*B(:,j)
                        // get C(m-4:m-1,j)
                        #ifdef GB_AVX512F
                        v4 c1 = (*((v4u *) (Cxj + m-4))) ;
                        #else
                        GB_CTYPE cx [4] ;
                        memcpy (cx, Cxj + m-4, 4 * sizeof (GB_CTYPE)) ;
                        #endif
                        // get A(m-4,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 4 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-4,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-4:m-1,j) += A(m-4:m-1,k)*B(k,j)
                            #ifdef GB_AVX512F
                            c1 += (*((v4u *) ax)) * bkj ;
                            #else
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                            cx [3] += ax [3] * bkj ;
                            #endif
                        }
                        // save C(m-4:m-1,j)
                        #ifdef GB_AVX512F
                        (*((v4u *) (Cxj + m-4))) = c1 ;
                        #else
                        memcpy (Cxj + m-4, cx, 4 * sizeof (GB_CTYPE)) ;
                        #endif
                    }
                    break ;

                case 3:
                    {
                        // C(m-3:m-1,j) += A(m-3:m-1,j)*B(:,j)
                        GB_CTYPE cx [3] ;
                        memcpy (cx, Cxj + m-3, 3 * sizeof (GB_CTYPE)) ;
                        // get A(m-3,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 3 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-3,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-3:m-1,j) += A(m-3:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                            cx [2] += ax [2] * bkj ;
                        }
                        memcpy (Cxj + m-3, cx, 3 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 2:
                    {
                        // C(m-2:m-1,j) += A(m-2:m-1,j)*B(:,j)
                        GB_CTYPE cx [2] ;
                        memcpy (cx, Cxj + m-2, 2 * sizeof (GB_CTYPE)) ;
                        // get A(m-2,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 2 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-2,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-2:m-1,j) += A(m-2:m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                            cx [1] += ax [1] * bkj ;
                        }
                        memcpy (Cxj + m-2, cx, 2 * sizeof (GB_CTYPE)) ;
                    }
                    break ;

                case 1:
                    {
                        // C(m-1,j) += A(m-1,j)*B(:,j)
                        GB_CTYPE cx [1] ;
                        cx [0]= Cxj [m-1] ;
                        // get A(m-1,0)
                        const GB_ATYPE *restrict Axm = Ax + m - 1 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            const GB_BTYPE bkj = Bx [pB] ;
                            // get A(m-1,k)
                            const GB_ATYPE *restrict ax = Axm + (k * m) ;
                            // C(m-1,j) += A(m-1,k)*B(k,j)
                            cx [0] += ax [0] * bkj ;
                        }
                        Cxj [m-1] = cx [0] ;
                    }
                    break ;

                default:
                    break ;
            }
        }
    }
}
    #undef N
    #undef I


#undef GB_CIJ_MULTADD

