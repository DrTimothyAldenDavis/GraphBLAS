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
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;

        //----------------------------------------------------------------------
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        //----------------------------------------------------------------------

        switch (m)
        {

            //------------------------------------------------------------------
            // C is 1-by-n
            //------------------------------------------------------------------

            case 1 :

                for (int64_t jB = jB_start ; jB < jB_end ; jB++)
                {
                    // get B(:,j) and C(:,j)
                    const int64_t j = GBH (Bh, jB) ;
                    const int64_t pB_start = Bp [jB] ;
                    const int64_t pB_end   = Bp [jB+1] ;
                    // load C(:,j)
                    GB_CTYPE cx [1] ;
                    cx [0] = Cx [j] ;
                    // C(:,j) += A*B(:,j)
                    GB_PRAGMA_SIMD
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        // bkj = B(k,j)
                        const int64_t k = Bi [pB] ;
                        GB_GETB (bkj, Bx, pB, B_iso) ;
                        // get A(:,k)
                        const int64_t pA = k ;
                        // C(:,j) += A(:,k)*B(k,j)
                        GB_CIJ_MULTADD (0,0) ; // C(0,j) += A(0,k)*B(k,j)
                    }
                    // save C(:,j)
                    Cx [j] = cx [0] ;
                }
                break ;

            //------------------------------------------------------------------
            // C is 2-by-n
            //------------------------------------------------------------------

            case 2 :

                for (int64_t jB = jB_start ; jB < jB_end ; jB++)
                {
                    // get B(:,j) and C(:,j)
                    const int64_t j = GBH (Bh, jB) ;
                    const int64_t pC = j << 1 ;
                    const int64_t pB_start = Bp [jB] ;
                    const int64_t pB_end   = Bp [jB+1] ;
                    // load C(:,j)
                    GB_CTYPE cx [2] ;
                    cx [0] = Cx [pC    ] ;
                    cx [1] = Cx [pC + 1] ;
                    // C(:,j) += A*B(:,j)
                    GB_PRAGMA_SIMD
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        // bkj = B(k,j)
                        const int64_t k = Bi [pB] ;
                        GB_GETB (bkj, Bx, pB, B_iso) ;
                        // get A(:,k)
                        const int64_t pA = k << 1 ;
                        // C(:,j) += A(:,k)*B(k,j)
                        GB_CIJ_MULTADD (0,0) ; // C(0,j) += A(0,k)*B(k,j)
                        GB_CIJ_MULTADD (1,1) ; // C(1,j) += A(1,k)*B(k,j)
                    }
                    // save C(:,j)
                    Cx [pC  ] = cx [0] ;
                    Cx [pC+1] = cx [1] ;
                }
                break ;

            //------------------------------------------------------------------
            // C is 3-by-n
            //------------------------------------------------------------------

            case 3 :

                for (int64_t jB = jB_start ; jB < jB_end ; jB++)
                {
                    // get B(:,j) and C(:,j)
                    const int64_t j = GBH (Bh, jB) ;
                    const int64_t pC = j * 3 ;
                    const int64_t pB_start = Bp [jB] ;
                    const int64_t pB_end   = Bp [jB+1] ;
                    // load C(:,j)
                    GB_CTYPE cx [3] ;
                    cx [0] = Cx [pC    ] ;    // cx [0] = C(0,j)
                    cx [1] = Cx [pC + 1] ;    // cx [1] = C(1,j)
                    cx [2] = Cx [pC + 2] ;    // cx [2] = C(2,j)
                    // C(:,j) += A*B(:,j)
                    GB_PRAGMA_SIMD
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        // bkj = B(k,j)
                        const int64_t k = Bi [pB] ;
                        GB_GETB (bkj, Bx, pB, B_iso) ;
                        // get A(:,k)
                        const int64_t pA = k * 3 ;
                        // C(:,j) += A(:,k)*B(k,j)
                        GB_CIJ_MULTADD (0,0) ; // C(0,j) += A(0,k)*B(k,j)
                        GB_CIJ_MULTADD (1,1) ; // C(1,j) += A(1,k)*B(k,j)
                        GB_CIJ_MULTADD (2,2) ; // C(2,j) += A(2,k)*B(k,j)
                    }
                    // save C(:,j)
                    Cx [pC  ] = cx [0] ;
                    Cx [pC+1] = cx [1] ;
                    Cx [pC+2] = cx [2] ;
                }
                break ;

            //------------------------------------------------------------------
            // C is 4-by-n
            //------------------------------------------------------------------

            case 4 :

                // C is 4-by-n
                for (int64_t jB = jB_start ; jB < jB_end ; jB++)
                {
                    // get B(:,j) and C(:,j)
                    const int64_t j = GBH (Bh, jB) ;
                    const int64_t pC = j << 2 ;
                    const int64_t pB_start = Bp [jB] ;
                    const int64_t pB_end   = Bp [jB+1] ;
                    // load C(:,j)
                    GB_CTYPE cx [4] ;
                    cx [0] = Cx [pC    ] ;    // cx [0] = C(0,j)
                    cx [1] = Cx [pC + 1] ;    // cx [1] = C(1,j)
                    cx [2] = Cx [pC + 2] ;    // cx [2] = C(2,j)
                    cx [3] = Cx [pC + 3] ;    // cx [3] = C(3,j)
                    // C(:,j) += A*B(:,j)
                    GB_PRAGMA_SIMD
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        // bkj = B(k,j)
                        const int64_t k = Bi [pB] ;
                        GB_GETB (bkj, Bx, pB, B_iso) ;
                        // get A(:,k)
                        const int64_t pA = k << 2 ;
                        GB_CIJ_MULTADD (0,0) ; // C(0,j) += A(0,k)*B(k,j)
                        GB_CIJ_MULTADD (1,1) ; // C(1,j) += A(1,k)*B(k,j)
                        GB_CIJ_MULTADD (2,2) ; // C(2,j) += A(2,k)*B(k,j)
                        GB_CIJ_MULTADD (3,3) ; // C(3,j) += A(3,k)*B(k,j)
                    }
                    // save C(:,j)
                    Cx [pC  ] = cx [0] ;
                    Cx [pC+1] = cx [1] ;
                    Cx [pC+2] = cx [2] ;
                    Cx [pC+3] = cx [3] ;
                }
                break ;

            //------------------------------------------------------------------
            // C is m-by-n where m > 4
            //------------------------------------------------------------------

            default :

                for (int64_t jB = jB_start ; jB < jB_end ; jB++)
                {
                    // get B(:,j) and C(:,j)
                    const int64_t j = GBH (Bh, jB) ;
                    const int64_t pC = j * m ;
                    const int64_t pB_start = Bp [jB] ;
                    const int64_t pB_end   = Bp [jB+1] ;
                    // C(:,j) += A*B(:,j), but only on sets of 4 rows of C
                    int64_t i ;
                    for (i = 0 ; i < m - 3 ; i += 4)
                    {
                        GB_CTYPE cx [4] ;
                        // load C(i:i+3,j)
                        cx [0] = Cx [pC + i  ] ;    // cx [0] = C(i+0,j)
                        cx [1] = Cx [pC + i+1] ;    // cx [1] = C(i+1,j)
                        cx [2] = Cx [pC + i+2] ;    // cx [2] = C(i+2,j)
                        cx [3] = Cx [pC + i+3] ;    // cx [3] = C(i+3,j)
                        // C(i:i+3,j) += A(i:i+3,:)*B(:,j)
                        GB_PRAGMA_SIMD
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        {
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(:,k)
                            const int64_t pA = (k * m) + i ;
                            // C(i:i+3,j) += A(i:i+3,k)*B(k,j)
                            GB_CIJ_MULTADD (0, i  ) ;
                            GB_CIJ_MULTADD (1, i+1) ;
                            GB_CIJ_MULTADD (2, i+2) ;
                            GB_CIJ_MULTADD (3, i+3) ;
                        }
                        // save C(i:i+3,j)
                        Cx [pC + i  ] = cx [0] ;
                        Cx [pC + i+1] = cx [1] ;
                        Cx [pC + i+2] = cx [2] ;
                        Cx [pC + i+3] = cx [3] ;
                    }

                    // cleanup, if m is not divisible by 4
                    switch (m - i)
                    {

                        case 3 :  
                            {
                                GB_CTYPE cx [3] ;
                                // load C(m-3:m-1,j)
                                cx [0] = Cx [pC + m-3] ;
                                cx [1] = Cx [pC + m-2] ;
                                cx [2] = Cx [pC + m-1] ;
                                // C(m-3:m-1,j) += A(m-3:m-1,:)*B(:,j)
                                GB_PRAGMA_SIMD
                                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                                {
                                    // bkj = B(k,j)
                                    const int64_t k = Bi [pB] ;
                                    GB_GETB (bkj, Bx, pB, B_iso) ;
                                    // get A(:,k)
                                    const int64_t pA = (k * m) + (m-3) ;
                                    // C(m-3:m-1,j) += A(m-3:m-1,k)*B(k,j)
                                    GB_CIJ_MULTADD (0, m-3) ;
                                    GB_CIJ_MULTADD (1, m-2) ;
                                    GB_CIJ_MULTADD (2, m-1) ;
                                }
                                // save C(:m-1,j)
                                Cx [pC + m-3] = cx [0] ;
                                Cx [pC + m-2] = cx [1] ;
                                Cx [pC + m-1] = cx [2] ;
                            }
                            break ;

                        case 2 :  
                            {
                                GB_CTYPE cx [2] ;
                                // load C(m-2:m-1,j)
                                cx [0] = Cx [pC + m-2] ;
                                cx [1] = Cx [pC + m-1] ;
                                // C(m-2:m-1,j) += A(m-2:m-1,:)*B(:,j)
                                GB_PRAGMA_SIMD
                                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                                {
                                    // bkj = B(k,j)
                                    const int64_t k = Bi [pB] ;
                                    GB_GETB (bkj, Bx, pB, B_iso) ;
                                    // get A(:,k)
                                    const int64_t pA = (k * m) + (m-2) ;
                                    // C(m-2:m-1,j) += A(m-2:m-1,k)*B(k,j)
                                    GB_CIJ_MULTADD (0, m-2) ;
                                    GB_CIJ_MULTADD (1, m-1) ;
                                }
                                // save C(m-2:m-1,j)
                                Cx [pC + m-2] = cx [0] ;
                                Cx [pC + m-1] = cx [1] ;
                            }
                            break ;

                        case 1 :  
                            {
                                GB_CTYPE cx [1] ;
                                // load C(m-1,j)
                                cx [0] = Cx [pC + m-1] ;
                                // C(m-1,j) += A(m-1,:)*B(:,j)
                                GB_PRAGMA_SIMD
                                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                                {
                                    // bkj = B(k,j)
                                    const int64_t k = Bi [pB] ;
                                    GB_GETB (bkj, Bx, pB, B_iso) ;
                                    // get A(:,k)
                                    const int64_t pA = (k * m) + (m-1) ;
                                    // C(m-1,j) += A(m-1,k)*B(k,j)
                                    GB_CIJ_MULTADD (0, m-1) ;
                                }
                                // save C(m-1,j)
                                Cx [pC + m-1] = cx [0] ;
                            }
                            break ;

                        default : 
                            break ;
                    }
                }
                break ;
        }
    }
}

#if 0
{

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;

        //----------------------------------------------------------------------

        for (int64_t jB = jB_start ; jB < jB_end ; jB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            const int64_t j = GBH (Bh, jB) ;
            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = Bp [jB] ;
            const int64_t pB_end   = Bp [jB+1] ;

            //------------------------------------------------------------------
            // C(:,j) += A*B(:,j)
            //------------------------------------------------------------------

            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            {
                // bkj = B(k,j)
                const int64_t k = Bi [pB] ;
                GB_GET_B_kj ;

                // get A(:,k)
                const int64_t pA_start = k * avlen ;

                //--------------------------------------------------------------
                // C(:,j) += A(:,k)*B(k,j)
                //--------------------------------------------------------------

                for (int64_t i = 0 ; i < cvlen ; i++)
                {
                    // C(i,j) += A(i,k)*B(k,j)
                    const int64_t pA = i + pA_start ;
                    #if A_IS_BITMAP
                    if (!Ab [pA]) continue ;
                    #endif
                    // aik = A(i,k)
                    GB_GETA (aik, Ax, pA, A_iso) ;
                    // C(i,j) += aik * bkj
                    GB_MULTADD (Cx [i + pC_start], aik, bkj, i, k, j) ;
                }
            }
        }
    }
}
#endif

#undef GB_CIJ_MULTADD

