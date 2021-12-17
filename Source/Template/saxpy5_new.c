//------------------------------------------------------------------------------
// GB_AxB_saxpy5_template.c: C+=A*B when C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is as-if-full.
// A is bitmap or full.
// B is sparse or hypersparse.

#if !A_IS_PATTERN
if (A_iso)
#endif
{

    //--------------------------------------------------------------------------
    // C += A*B where A is bitmap/full, and either iso-valued or pattern-only
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        #if !GB_A_IS_PATTERN
        // get the iso value of A
        const GB_ATYPE ax = Ax [0] ;
        #endif
        // get the task descriptor
        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start ; jB < jB_end ; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBH (Bh, jB) ;
            const int64_t pC = j * m ;
            const int64_t pB_start = Bp [jB] ;
            const int64_t pB_end   = Bp [jB+1] ;
            // C(:,j) += A*B(:,j)
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            {
                // get B(k,j)
                const int64_t k = Bi [pB] ;
                #if A_IS_BITMAP
                // get A(:,k)
                const int64_t pA = k * m ;
                #endif
                #if GB_IS_FIRSTI_MULTIPLIER
                {
                    for (int64_t i = 0 ; i < m ; i++)
                    { 
                        #if A_IS_BITMAP
                        if (!Ab [pA + i]) continue ;
                        #endif
                        // C(i,j) += (i + GB_OFFSET) ;
                        GB_CIJ_UPDATE (pC + i, i + GB_OFFSET) ;
                    }
                }
                #else
                {
                    // t = ax * bkj
                    GB_CTYPE t ;
                    GB_MULT (t, ax, GBX (Bx, pB, B_iso), ignore, k, j) ;
                    // C(:,j) += t
                    for (int64_t i = 0 ; i < m ; i++)
                    { 
                        #if A_IS_BITMAP
                        if (!Ab [pA + i]) continue ;
                        #endif
                        // C(i,j) += t ;
                        GB_CIJ_UPDATE (pC + i, t) ;
                    }
                }
                #endif
            }
        }
    }

}
#if !GB_A_IS_PATTERN
else
{

    #if A_IS_BITMAP
    {

        //--------------------------------------------------------------------------
        // C += A*B where A is bitmap (and not iso or pattern-only)
        //--------------------------------------------------------------------------

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
                const int64_t pC = j * m ;
                const int64_t pB_start = Bp [jB] ;
                const int64_t pB_end   = Bp [jB+1] ;
                // C(:,j) += A*B(:,j)
                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                { 
                    // get B(k,j)
                    const int64_t k = Bi [pB] ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    // get A(:,k)
                    const int64_t pA = k * m ;
                    // C(:,j) += A(:,k)*B(k,j)
                    for (int64_t i = 0 ; i < m ; i++)
                    { 
                        if (!Ab [pA+i]) continue ;
                        // C(i,j) += A(i,k)*B(k,j) ;
                        GB_MULTADD (Cx [pC + i], Ax [pA + i], bkj, i, k, j) ;
                    }
                }
            }
        }

    }
    #else
    {

        //--------------------------------------------------------------------------
        // C += A*B where A is full (and not iso or pattern-only)
        //--------------------------------------------------------------------------

        // GB_CIJ_MULTADD:  C(i,j) += A(i,k) * B(k,j)
        // the semiring is not positional (or A would be pattern-only), so the
        // i, k, j values are not needed
        #define GB_CIJ_MULTADD(cij,aik,bkj) \
            GB_MULTADD (cij, aik, bkj, ignore, ignore, ignore) ;

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

                //------------------------------------------------------------------
                // C(:,j) += A*B(:,j), on sets of 16 rows of C and A at a time
                //------------------------------------------------------------------

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
                        GB_GETB (bkj, Bx, pB, B_iso) ;
                        // get A(i,k)
                        const GB_ATYPE *restrict ax = Axi + (k * m) ;
                        // C(i:i+15,j) += A(i:i+15,k)*B(k,j)
                        #ifdef GB_AVX512F
                        GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                        GB_CIJ_MULTADD (c2, (*((v8u *) (ax + 8))), bkj) ;
                        #else
                        GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                        GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                        GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                        GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                        GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                        GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                        GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                        GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                        GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                        GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                        GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                        GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                        GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                        GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                        GB_CIJ_MULTADD (cx [14], ax [14], bkj) ;
                        GB_CIJ_MULTADD (cx [15], ax [15], bkj) ;
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

                //------------------------------------------------------------------
                // C(m-N:m-1,j) += A(m-N:m-1,j)*B(:,j) for last 0 to 15 rows
                //------------------------------------------------------------------

                switch (m & 15)
                {

                    //--------------------------------------------------------------
                    // C(m-15:m-1,j) += A(m-15:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 15:
                        {
                            // load C(m-15:m-1,j)
                            GB_CTYPE cx [15] ;
                            memcpy (cx, Cxj + m-15, 15 * sizeof (GB_CTYPE)) ;
                            // get A(m-15,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 15 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-15,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-15:m-1,j) += A(m-15:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                                GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                                GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                                GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                                GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                                GB_CIJ_MULTADD (cx [14], ax [14], bkj) ;
                            }
                            // save C(m-15:m-1,j)
                            memcpy (Cxj + m-15, cx, 15 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-14:m-1,j) += A(m-14:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 14:
                        {
                            // load C(m-14:m-1,j)
                            GB_CTYPE cx [14] ;
                            memcpy (cx, Cxj + m-14, 14 * sizeof (GB_CTYPE)) ;
                            // get A(m-14,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 14 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-14,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-14:m-1,j) += A(m-14:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                                GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                                GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                                GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                                GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                            }
                            // save C(m-14:m-1,j)
                            memcpy (Cxj + m-14, cx, 14 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-13:m-1,j) += A(m-13:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 13:
                        {
                            // load C(m-13:m-1,j)
                            GB_CTYPE cx [13] ;
                            memcpy (cx, Cxj + m-13, 13 * sizeof (GB_CTYPE)) ;
                            // get A(m-13,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 13 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-13,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-13:m-1,j) += A(m-13:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                                GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                                GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                                GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                            }
                            // save C(m-13:m-1,j)
                            memcpy (Cxj + m-13, cx, 13 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-12:m-1,j) += A(m-12:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 12:
                        {
                            // C(m-12:m-1,j) += A(m-12:m-1,j)*B(:,j)
                            // load C(m-12:m-1,j)
                            GB_CTYPE cx [12] ;
                            memcpy (cx, Cxj + m-12, 12 * sizeof (GB_CTYPE)) ;
                            // get A(m-12,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 12 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-12,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-12:m-1,j) += A(m-12:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                                GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                                GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                            }
                            // save C(m-12:m-1,j)
                            memcpy (Cxj + m-12, cx, 12 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-11:m-1,j) += A(m-11:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 11:
                        {
                            // load C(m-11:m-1,j)
                            GB_CTYPE cx [11] ;
                            memcpy (cx, Cxj + m-11, 11 * sizeof (GB_CTYPE)) ;
                            // get A(m-11,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 11 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-11,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-11:m-1,j) += A(m-11:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                                GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            }
                            // save C(m-11:m-1,j)
                            memcpy (Cxj + m-11, cx, 11 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-10:m-1,j) += A(m-10:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 10:
                        {
                            // load C(m-10:m-1,j)
                            GB_CTYPE cx [10] ;
                            memcpy (cx, Cxj + m-10, 10 * sizeof (GB_CTYPE)) ;
                            // get A(m-10,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 10 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-10,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-10:m-1,j) += A(m-10:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                                GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            }
                            // save C(m-10:m-1,j)
                            memcpy (Cxj + m-10, cx, 10 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-9:m-1,j) += A(m-9:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 9:
                        {
                            // load C(m-9:m-1,j)
                            GB_CTYPE cx [9] ;
                            memcpy (cx, Cxj + m-9, 9 * sizeof (GB_CTYPE)) ;
                            // get A(m-9,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 9 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-9,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-9:m-1,j) += A(m-9:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                                GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            }
                            // save C(m-9:m-1,j)
                            memcpy (Cxj + m-9, cx, 9 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-8:m-1,j) += A(m-8:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 8:
                        {
                            // load C(m-8:m-1,j)
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
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-8,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-8:m-1,j) += A(m-8:m-1,k)*B(k,j)
                                #ifdef GB_AVX512F
                                GB_CIJ_MULTADD (c1, (*((v8u *) ax)), bkj) ;
                                #else
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                                GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
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

                    //--------------------------------------------------------------
                    // C(m-7:m-1,j) += A(m-7:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 7:
                        {
                            // load C(m-7:m-1,j)
                            GB_CTYPE cx [7] ;
                            memcpy (cx, Cxj + m-7, 7 * sizeof (GB_CTYPE)) ;
                            // get A(m-7,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 7 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-7,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-7:m-1,j) += A(m-7:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                                GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            }
                            // save C(m-7:m-1,j)
                            memcpy (Cxj + m-7, cx, 7 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-6:m-1,j) += A(m-6:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 6:
                        {
                            // load C(m-6:m-1,j)
                            GB_CTYPE cx [6] ;
                            memcpy (cx, Cxj + m-6, 6 * sizeof (GB_CTYPE)) ;
                            // get A(m-6,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 6 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-6,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-6:m-1,j) += A(m-6:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                                GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            }
                            // save C(m-6:m-1,j)
                            memcpy (Cxj + m-6, cx, 6 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-5:m-1,j) += A(m-5:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 5:
                        {
                            // load C(m-5:m-1,j)
                            GB_CTYPE cx [5] ;
                            memcpy (cx, Cxj + m-5, 5 * sizeof (GB_CTYPE)) ;
                            // get A(m-5,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 5 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-5,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-5:m-1,j) += A(m-5:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                                GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            }
                            // save C(m-5:m-1,j)
                            memcpy (Cxj + m-5, cx, 5 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-4:m-1,j) += A(m-4:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 4:
                        {
                            // load C(m-4:m-1,j)
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
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-4,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-4:m-1,j) += A(m-4:m-1,k)*B(k,j)
                                #ifdef GB_AVX512F
                                GB_CIJ_MULTADD (c1, (*((v4u *) ax)), bkj) ;
                                #else
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                                GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
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

                    //--------------------------------------------------------------
                    // C(m-3:m-1,j) += A(m-3:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 3:
                        {
                            // load C(m-3:m-1,j)
                            GB_CTYPE cx [3] ;
                            memcpy (cx, Cxj + m-3, 3 * sizeof (GB_CTYPE)) ;
                            // get A(m-3,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 3 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-3,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-3:m-1,j) += A(m-3:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                                GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            }
                            // save C(m-3:m-1,j)
                            memcpy (Cxj + m-3, cx, 3 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-2:m-1,j) += A(m-2:m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 2:
                        {
                            // load C(m-2:m-1,j)
                            GB_CTYPE cx [2] ;
                            memcpy (cx, Cxj + m-2, 2 * sizeof (GB_CTYPE)) ;
                            // get A(m-2,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 2 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-2,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-2:m-1,j) += A(m-2:m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                                GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            }
                            // save C(m-2:m-1,j)
                            memcpy (Cxj + m-2, cx, 2 * sizeof (GB_CTYPE)) ;
                        }
                        break ;

                    //--------------------------------------------------------------
                    // C(m-1,j) += A(m-1,j)*B(:,j)
                    //--------------------------------------------------------------

                    case 1:
                        {
                            // load C(m-1,j)
                            GB_CTYPE cx [1] ;
                            cx [0]= Cxj [m-1] ;
                            // get A(m-1,0)
                            const GB_ATYPE *restrict Axm = Ax + m - 1 ;
                            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                            { 
                                // bkj = B(k,j)
                                const int64_t k = Bi [pB] ;
                                GB_GETB (bkj, Bx, pB, B_iso) ;
                                // get A(m-1,k)
                                const GB_ATYPE *restrict ax = Axm + (k * m) ;
                                // C(m-1,j) += A(m-1,k)*B(k,j)
                                GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            }
                            // save C(m-1,j)
                            Cxj [m-1] = cx [0] ;
                        }
                        break ;

                    default:
                        break ;
                }
            }
        }
    }
    #endif
}
#endif

#undef GB_CIJ_MULTADD

