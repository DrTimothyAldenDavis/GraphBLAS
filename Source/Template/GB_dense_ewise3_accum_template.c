//------------------------------------------------------------------------------
// GB_dense_ewise3_accum_template: C += A+B where all 3 matrices are dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_ATYPE *Ax = A->x ;
    GB_BTYPE *Bx = B->x ;
    GB_CTYPE *Cx = C->x ;
    const int64_t cnz = GB_NNZ (C) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C += A+B where all 3 matries are dense
    //--------------------------------------------------------------------------

    if (A == B)
    {

        //----------------------------------------------------------------------
        // C += 2*A where A and C are dense
        //----------------------------------------------------------------------

        #if GB_HAS_CBLAS & GB_OP_IS_PLUS_REAL

            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 2, Ax, Cx, nthreads) ;   // C += 2*A

        #else

            // C += A+A
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GETA (aij, Ax, p) ;                  // aij = Ax [p]
                GB_CTYPE_SCALAR (t) ;                   // declare scalar t
                GB_BINOP (t, aij, aij) ;                // t = aij + aij
                GB_BINOP (GB_CX (p), GB_CX (p), t) ;    // Cx [p] = cij + t
            }

        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // C += A+B where all 3 matrices are dense
        //----------------------------------------------------------------------

        #if GB_HAS_CBLAS & GB_OP_IS_PLUS_REAL

            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 1, Ax, Cx, nthreads) ;   // C += A
            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 1, Bx, Cx, nthreads) ;   // C += B

        #else

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GETA (aij, Ax, p) ;                  // aij = Ax [p]
                GB_GETB (bij, Bx, p) ;                  // bij = Bx [p]
                GB_CTYPE_SCALAR (t) ;                   // declare scalar t
                GB_BINOP (t, aij, bij) ;                // t = aij + bij
                GB_BINOP (GB_CX (p), GB_CX (p), t) ;    // Cx [p] = cij + t
            }

        #endif
    }
}

