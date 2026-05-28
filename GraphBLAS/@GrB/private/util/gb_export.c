//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a GraphBLAS C.opaque @GrB handle
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gb_export (&C_opaque, &C, kind): exports C as a MATLAB 8-byte C.opaque uint8
// mxArray, containing a single pointer to a GrB_Matrix.  The input GrB_Matrix
// C may be shallow or deep.

// No mx* methods are called, so that any memory allocation failures can
// be properly handled.

#define GB_UTIL

#define FREE_WORK                   \
    GrB_Matrix_free (&T) ;

#define FREE_ALL                    \
    GrB_Matrix_free (C_handle) ;

#include "gb_interface.h"

GrB_Info gb_export              // export a GrB_Matrix to MATLAB
(
    // output:
    GrB_Matrix *C_opaque,
    // input/output:
    GrB_Matrix *C_handle,       // GrB_Matrix to export, set to NULL on output
    // input:
    kind_enum_t kind            // GrB, sparse, full, or built-in
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, T = NULL ;
    CHECK_ERROR (C_handle == NULL || (*C_handle == NULL), "internal error 3") ;
    C = (*C_handle) ;

    // OK (GxB_Matrix_fprint (C, "at start of gb_export", 5, NULL)) ;

    //--------------------------------------------------------------------------
    // ensure C has no readonly components
    //--------------------------------------------------------------------------

    int readonly ;
    OK (GrB_Matrix_get_INT32 (C, &readonly, GxB_IS_READONLY)) ;

    if (readonly)
    { 
        // C has readonly components so make a deep copy
        OK (GrB_Matrix_dup (&T, C)) ;
        GrB_Matrix_free (C_handle) ;
        (*C_handle) = T ;
        T = NULL ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // determine if all entries in C are present
    //--------------------------------------------------------------------------

    if (kind == KIND_BUILTIN)
    { 
        // export as full if all entries present, or sparse otherwise
        uint64_t nrows, ncols, nvals ;
        OK (GrB_Matrix_nvals (&nvals, C)) ;
        OK (GrB_Matrix_nrows (&nrows, C)) ;
        OK (GrB_Matrix_ncols (&ncols, C)) ;
        bool is_full = ((double) nrows * (double) ncols == (double) nvals) ;
        kind = (is_full) ? KIND_FULL : KIND_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // conform the matrix to a MATLAB sparse or full format, if requested
    //--------------------------------------------------------------------------

    if (kind == KIND_SPARSE)
    { 

        //----------------------------------------------------------------------
        // export C as a @GrB matrix, to become a MATLAB sparse matrix
        //----------------------------------------------------------------------

        // Typecast to double, if C is integer (int8, ..., uint64)
        OK (gb_export_to_sparse (C_handle)) ;
        C = (*C_handle) ;

    }
    else if (kind == KIND_FULL)
    { 

        //----------------------------------------------------------------------
        // export C as a @GrB matrix, to become a MATLAB full matrix
        //----------------------------------------------------------------------

        OK (gb_export_to_full (C_handle)) ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // copy the handle into C_opaque and return result
    //--------------------------------------------------------------------------

    // C should now be deep, but double-check here
    OK (GrB_Matrix_get_INT32 (C, &readonly, GxB_IS_READONLY)) ;
    CHECK_ERROR (readonly, "internal error 7") ;

    (*C_opaque) = C ;       // copy the GraphBLAS C header into C_opaque
    (*C_handle) = NULL ;    // flag C as no longer available to the caller
    return (GrB_SUCCESS) ;
}

