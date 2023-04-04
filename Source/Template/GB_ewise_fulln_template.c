//------------------------------------------------------------------------------
// GB_ewise_fulln_template: C = A+B where all 3 matrices are full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_unused.h"

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    GB_B_TYPE *Bx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *Cx = (GB_C_TYPE *) C->x ;
    const int64_t cnz = GB_nnz (C) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (B)) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!C->iso) ;
    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C = A+B where all 3 matrices are full
    //--------------------------------------------------------------------------

    #if GB_CTYPE_IS_BTYPE

    if (C == B)
    {

        //----------------------------------------------------------------------
        // C = A+C where A and C are full
        //----------------------------------------------------------------------

        // C and B cannot be aliased if their types differ
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;                // aij = Ax [p]
            GB_BINOP (GB_CX (p), aij, GB_CX (p), 0, 0) ; // Cx [p] = aij+Cx [p]
        }

    }
    else 
    #endif

    #if GB_CTYPE_IS_ATYPE

    if (C == A)
    {

        //----------------------------------------------------------------------
        // C = C+B where B and C are full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            GB_DECLAREB (bij) ;
            GB_GETB (bij, Bx, p, false) ;                   // bij = Bx [p]
            GB_BINOP (GB_CX (p), GB_CX (p), bij, 0, 0) ;    // Cx [p] += bij
        }

    }
    else
    #endif

    {

        //----------------------------------------------------------------------
        // C = A+B where all 3 matrices are full
        //----------------------------------------------------------------------

        // note that A and B may still be aliased to each other
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;               // aij = Ax [p]
            GB_DECLAREB (bij) ;
            GB_GETB (bij, Bx, p, false) ;               // bij = Bx [p]
            GB_BINOP (GB_CX (p), aij, bij, 0, 0) ;      // Cx [p] = aij + bij
        }
    }
}

