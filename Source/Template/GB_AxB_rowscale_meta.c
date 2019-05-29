//------------------------------------------------------------------------------
// GB_AxB_rowscale_meta: C=D*B where D is a square diagonal matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// All vectors C=D*B are computed fully in parallel. 

// PARALLEL: done, but use GB_ewise_slice for better parallelism.

{

    //--------------------------------------------------------------------------
    // get C, D, and B
    //--------------------------------------------------------------------------

    const GB_ATYPE *restrict Dx = D_is_pattern ? NULL : D->x ;
    const GB_BTYPE *restrict Bx = B_is_pattern ? NULL : B->x ;
    const int64_t  *restrict Bi = B->i ;

    //--------------------------------------------------------------------------
    // C=D*B
    //--------------------------------------------------------------------------

    GBI_parallel_for_each_vector (B, nthreads)
    {

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        GBI_jth_iteration (j, pB, pB_end) ;

        //----------------------------------------------------------------------
        // C(:,j) = D*B(:,j)
        //----------------------------------------------------------------------

        // #pragma omp simd
        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = Bi [pB] ;
            GB_GETA (dii, Dx, i) ;
            GB_GETB (bij, Bx, pB) ;
            GB_BINOP (GB_CX (pB), dii, bij) ;
        }
    }
}

