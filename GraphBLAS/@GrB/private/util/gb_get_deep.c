//------------------------------------------------------------------------------
// gb_get_deep: create a deep GrB_Matrix from a MATLAB input matrix/object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns a deep copy GrB_Matrix C for a @GrB or MATLAB input matrix, with no
// pending work.  Used for methods such as
//
//      C = GrB.apply (Cin, ... )
//
// where Cin is either a MATLAB matrix, or a @GrB object that must not be
// modified (except any pending work is finished in Cin).  The caller does not
// need Cin, just its deep copy C, which the caller will then modify and
// return as pargout [0].  Thus Cin is not returned to the caller.

// This method is not used for the in-place syntax:
//
//      GrB.apply (C, ... )
//
// with nargout = 0, since in this case, C is modified in place (and it must
// also be a @GrB object, not a MATLAB matrix).

#define GB_UTIL
#define FREE_ALL                    \
    GrB_Matrix_free (&C) ;          \
    GrB_Matrix_free (C_shallow) ;

#include "gb_interface.h"

GrB_Info gb_get_deep        // get a deep GrB_Matrix copy of a matrix
(
    // output:
    GrB_Matrix *C_handle,   // deep copy of the input matrix
    GrB_Matrix *C_shallow,  // shallow version; must be freed by caller
    // input:
    gb_matrix X             // input MATLAB or @GrB matrix
)
{ 

    //--------------------------------------------------------------------------
    // get the GrB_Matrix Cin and optional C_shallow of a MATLAB matrix
    //--------------------------------------------------------------------------

    GrB_Matrix Cin = NULL, C = NULL ;
    OK (gb_get_matrix (&Cin, C_shallow, X)) ;

    //--------------------------------------------------------------------------
    // ensure Cin has no pending work
    //--------------------------------------------------------------------------

    // a MATLAB matrix has no pending work; only check if X is a @GrB object
    if (X->G != NULL)
    { 
        OK (GrB_Matrix_wait (Cin, GrB_MATERIALIZE)) ;
    }

    //--------------------------------------------------------------------------
    // make a deep copy of Cin, typecasting from a MATLAB matrix if needed
    //--------------------------------------------------------------------------

    int fmt ;   // by row or by column
    OK (GrB_Matrix_get_INT32 (Cin, &fmt, GxB_FORMAT)) ;
    OK (gb_typecast (&C, Cin, NULL, fmt, 0)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*C_handle) = C ;
    return (GrB_SUCCESS) ;
}

