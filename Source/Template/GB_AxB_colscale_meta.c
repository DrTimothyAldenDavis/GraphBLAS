//------------------------------------------------------------------------------
// GB_AxB_colscale_meta: C=A*D where D is a square diagonal matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// PARALLEL: all vectors C=A*D are computed fully in parallel. 

{

    //--------------------------------------------------------------------------
    // get C, A, and D
    //--------------------------------------------------------------------------

    const GB_ATYPE *restrict Ax = A_is_pattern ? NULL : A->x ;
    const GB_BTYPE *restrict Dx = D_is_pattern ? NULL : D->x ;

    //--------------------------------------------------------------------------
    // C=A*D
    //--------------------------------------------------------------------------

    GBI_parallel_for_each_vector (A, nthreads)
    {

        //----------------------------------------------------------------------
        // get A(:,j)
        //----------------------------------------------------------------------

        GBI_jth_iteration (j, pA, pA_end) ;
        int64_t ajnz = pA_end - pA ;
        // no work to do if A(:,j) is empty
        if (ajnz == 0) continue ;

        //----------------------------------------------------------------------
        // C(:,j) = A(:,j)*D(j,j)
        //----------------------------------------------------------------------

        GB_GETB (djj, Dx, j) ;

        #pragma omp simd
        for ( ; pA < pA_end ; pA++)
        {
            GB_GETA (aij, Ax, pA) ;
            GB_BINOP (GB_CX (pA), aij, djj) ;
        }
    }
}

