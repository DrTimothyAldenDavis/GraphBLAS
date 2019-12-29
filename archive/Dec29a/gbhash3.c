//------------------------------------------------------------------------------
// gbhash3: sparse matrix-matrix multiplication using hash method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C = GrB.hash3 (A, B)

// A and B must be double (GrB_FP64)

#include "gb_matlab.h"
#include "myhash.h"

#define USAGE "usage: C = GrB.hash3 (A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 3 && nargout == 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    GrB_Descriptor desc = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt, &base) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Matrix A, B ;

    A = gb_get_shallow (pargin [0]) ;
    B = gb_get_shallow (pargin [1]) ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;
    CHECK_ERROR (type != GrB_FP64, "A must be double") ;

    OK (GxB_Matrix_type (&type, B)) ;
    CHECK_ERROR (type != GrB_FP64, "B must be double") ;

    GxB_Format_Value afmt ;
    OK (GxB_get (A, GxB_FORMAT, &afmt)) ;
    CHECK_ERROR (afmt != GxB_BY_COL, "A must be CSC") ;

    OK (GxB_get (B, GxB_FORMAT, &afmt)) ;
    CHECK_ERROR (afmt != GxB_BY_COL, "B must be CSC") ;

    bool is_hyper ;
    OK (GxB_get (A, GxB_IS_HYPER, &is_hyper)) ;
    CHECK_ERROR (is_hyper, "A must be standard, not hypersparse") ;

    OK (GxB_get (B, GxB_IS_HYPER, &is_hyper)) ;
    CHECK_ERROR (is_hyper, "A must be standard, not hypersparse") ;

    //--------------------------------------------------------------------------
    // get the content of A and B
    //--------------------------------------------------------------------------

    int64_t *Ap = A->p ;
    int64_t *Ai = A->i ;
    double  *Ax = A->x ;
    int64_t anrows, ancols ;

    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    int64_t *Bp = B->p ;
    int64_t *Bi = B->i ;
    double  *Bx = B->x ;
    int64_t bnrows, bncols ;

    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;

//    printf ("%g %g %g %g\n", 
//        (double) anrows, (double) ancols,
//        (double) bnrows, (double) bncols) ;

    CHECK_ERROR (ancols != bnrows, "dimensions wrong") ;

    //--------------------------------------------------------------------------
    // compute C = A*B
    //--------------------------------------------------------------------------

    int64_t *Cp = NULL ;
    int64_t *Ci = NULL ;
    double  *Cx = NULL ;
    int64_t cnrows = anrows, cncols = bncols ;
    int64_t nonempty = 0 ;

    myhash3 (&Cp, &Ci, &Cx,
        Ap, Ai, Ax, anrows, ancols,
        Bp, Bi, Bx, bnrows, bncols, &nonempty) ;

    int64_t cnvals = (Cp == NULL) ? 0 : Cp [cncols] ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    // double tic [2] ;
    // simple_tic (tic) ;

    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_import_CSC (&C, GrB_FP64, cnrows, cncols, cnvals, nonempty,
        &Cp, &Ci, (void **) (&Cx), NULL)) ;

    // double t1 = simple_toc (tic) ;
    // printf ("import time %g\n", t1) ;
    // simple_tic (tic) ;

    pargout [0] = gb_export (&C, kind) ;
    // double t2 = simple_toc (tic) ;
    // printf ("export time %g\n", t2) ;
    // simple_tic (tic) ;

    GB_WRAPUP ;

    // double t = simple_toc (tic) ;
    // printf ("wrapup time %g\n", t) ;
}

