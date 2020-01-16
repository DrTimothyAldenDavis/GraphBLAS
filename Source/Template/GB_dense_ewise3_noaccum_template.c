//------------------------------------------------------------------------------
// GB_dense_ewise3_noaccum_template: C = A+B where all 3 matrices are dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    GB_ATYPE *GB_RESTRICT Ax = A->x ;
    GB_BTYPE *GB_RESTRICT Bx = B->x ;
    GB_CTYPE *GB_RESTRICT Cx = C->x ;
    const int64_t cnz = GB_NNZ (C) ;

    //--------------------------------------------------------------------------
    // C += x where C is dense and x is a scalar
    //--------------------------------------------------------------------------

    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        GB_GETA (aij, Ax, p) ;                  // aij = Ax [p]
        GB_GETB (bij, Bx, p) ;                  // bij = Bx [p]
        GB_BINOP (GB_CX (p), aij, bij) ;        // Cx [p] = aij + bij
    }
}

