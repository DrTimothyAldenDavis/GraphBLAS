//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/grow_demo.c: grow a matrix row-by-row
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reads in a matrix A, then does C = A one row at a time.

#include "graphblas_demos.h"
#include "simple_rand.c"
#include "usercomplex.h"
#include "usercomplex.c"
#include "wathen.c"
#include "get_matrix.c"
#include "random_matrix.c"
#include "import_test.c"
#include "read_matrix.c"

// macro used by OK(...) to free workspace if an error occurs
#undef  FREE_ALL
#define FREE_ALL                            \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&C) ;                  \
    GrB_Vector_free (&w) ;                  \
    GrB_finalize ( ) ;

int main (int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, C = NULL ;
    GrB_Vector w = NULL ;
    GrB_Info info ;

    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "grow demo: nthreads %d\n", nthreads) ;

    //--------------------------------------------------------------------------
    // get A matrix
    //--------------------------------------------------------------------------

    OK (get_matrix (&A, argc, argv, false, false, false)) ;
    GrB_Index anrows, ancols, anvals ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;

    int type_code ;
    OK (GrB_Matrix_get_INT32 (A, &type_code, GrB_EL_TYPE_CODE)) ;
    GrB_Type atype = NULL ;
    OK (GxB_print (A, 5)) ;
    printf ("type_code: %d\n", type_code) ;

    switch (type_code)
    {
        case GrB_BOOL_CODE   : atype = GrB_BOOL    ; break ;
        case GrB_INT8_CODE   : atype = GrB_INT8    ; break ;
        case GrB_UINT8_CODE  : atype = GrB_UINT8   ; break ;
        case GrB_INT16_CODE  : atype = GrB_INT16   ; break ;
        case GrB_UINT16_CODE : atype = GrB_UINT16  ; break ;
        case GrB_INT32_CODE  : atype = GrB_INT32   ; break ;
        case GrB_UINT32_CODE : atype = GrB_UINT32  ; break ;
        case GrB_INT64_CODE  : atype = GrB_INT64   ; break ;
        case GrB_UINT64_CODE : atype = GrB_UINT64  ; break ;
        case GrB_FP32_CODE   : atype = GrB_FP32    ; break ;
        case GrB_FP64_CODE   : atype = GrB_FP64    ; break ;
        case GxB_FC32_CODE   : atype = GxB_FC32    ; break ;
        case GxB_FC64_CODE   : atype = GxB_FC64    ; break ;
        default              : ;
    }

    OK (GxB_print (atype, 5)) ;
    CHECK (atype != NULL, GrB_INVALID_VALUE) ;
    OK (GxB_print (A, 2)) ;

    //--------------------------------------------------------------------------
    // C = A, one row at a time
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_Vector_new (&w, atype, ancols)) ;
    OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;
    OK (GrB_set (C, false, GxB_HYPER_HASH)) ;
    OK (GrB_set (C, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (w, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    printf ("\n\nC empty:\n") ;
    OK (GxB_print (C, 2)) ;

    for (int64_t i = 0 ; i < anrows ; i++)
    {
        printf ("\n\ni = %ld\n", i) ;

        // w = A (i,:)
        OK (GrB_Col_extract (w, NULL, NULL, A, GrB_ALL, ancols, i,
            GrB_DESC_T0)) ;
        // OK (GxB_print (w, 2)) ;

        // C (i,:) = w
        OK (GrB_Row_assign (C, NULL, NULL, w, i, GrB_ALL, ancols, NULL)) ;

        // ensure C is finished
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
    }

    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;
    OK (GxB_print (C, 2)) ;
    FREE_ALL ;
    return (0) ;
}

