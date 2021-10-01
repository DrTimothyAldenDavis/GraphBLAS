//------------------------------------------------------------------------------
// GrB_Matrix_export: export a matrix in CSR, CSC, FullC, FullR, or COO format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Exports the contents of a matrix in one of 5 formats: CSR, CSC, FullC,
// FullR, or COO (triplet format).  The exported matrix is not modified.
// No typecast is performed; the output array Ax must be of the same type as
// the input matrix A.  This condition cannot be checked, and behavior is
// undefined if this condition does not hold.

// The required sizes of the Ap, Ai, and Ax arrays are given by
// GrB_Matrix_exportSize.

// The GraphBLAS C API does not have a GrB* method to query the type of a
// GrB_Matrix or the size of a type.  SuiteSparse:GraphBLAS provides
// GxB_Matrix_type_name to query the type of a matrix (returning a string),
// which can be converted into a GrB_Type with GxB_Type_from_name.  The size of
// a type can be queried with GxB_Type_size.  Using these methods, a user
// application can ensure that its Ax array has the correct size for any
// given GrB_Matrix it wishes to export, regardless of its type.

#include "GB_transpose.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_phbix_free (T) ;             \
}

GrB_Info GrB_Matrix_export  // export a matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, row indices for COO
    GrB_Index *Ai,          // row indices for CSR, CSC, col indices for COO
    void *Ax,               // values (must match the type of A_input)
    GrB_Format format,      // export format
    GrB_Matrix A_input      // matrix to export
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_export (Ap, Ai, Ax, format, A)") ;
    GB_BURBLE_START ("GrB_Matrix_export") ;

    GrB_Info info ;
    GrB_Matrix A = A_input ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = GB_clear_static_header (&T_header) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    switch (format)
    {
        case GrB_CSR_FORMAT :
        case GrB_CSC_FORMAT :
        case GrB_COO_FORMAT :
            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (Ai) ;
        default:
            GB_RETURN_IF_NULL (Ax) ;
    }

    // finish any pending work
    GB_MATRIX_WAIT (A) ;

    //--------------------------------------------------------------------------
    // determine current format of A and if a copy is needed
    //--------------------------------------------------------------------------

    int sparsity = GB_sparsity (A) ;
    bool is_csc = A->is_csc ;
    bool make_copy ;
    bool csc_requested ;

    switch (format)
    {
        case GrB_CSR_FORMAT :
            make_copy = !(sparsity == GxB_SPARSE && !is_csc) ;
            csc_requested = false ;
            break ;

        case GrB_CSC_FORMAT :
            make_copy = !(sparsity == GxB_SPARSE && is_csc) ;
            csc_requested = true ;
            break ;

        case GrB_DENSE_ROW_FORMAT :
            if (!GB_is_dense (A))
            { 
                // A must dense or full
                return (GrB_INVALID_VALUE) ;
            }
            make_copy = !(sparsity == GxB_FULL && !is_csc) ;
            csc_requested = false ;
            break ;

        case GrB_DENSE_COL_FORMAT :
            if (!GB_is_dense (A))
            { 
                // A must dense or full
                return (GrB_INVALID_VALUE) ;
            }
            make_copy = !(sparsity == GxB_FULL && is_csc) ;
            csc_requested = true ;
            break ;

        case GrB_COO_FORMAT : 
            // never make a copy to export in tuple format
            make_copy = false ;
            csc_requested = is_csc ;
            break ;

        default :
            // unknown format
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create a copy if the matrix is not in the requested format
    //--------------------------------------------------------------------------

    if (make_copy)
    { 
        if (is_csc != csc_requested)
        { 
            // T = A'
            GB_OK (GB_transpose_cast (T, A->type, csc_requested, A, false,
                Context)) ;
        }
        else
        { 
            // T = A
            GB_OK (GB_dup_worker (&T, false, A, true, A->type, Context)) ;
        }

        switch (format)
        {
            case GrB_CSR_FORMAT :
            case GrB_CSC_FORMAT :
                GB_OK (GB_convert_any_to_sparse (T, Context)) ;
                break ;

            case GrB_DENSE_ROW_FORMAT :
            case GrB_DENSE_COL_FORMAT :
                GB_convert_any_to_full (T) ;
                break ;

            default :
                break ;
        }
        A = T ;
    }

    //--------------------------------------------------------------------------
    // export the contents of the matrix
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    GrB_Index nvals = GB_nnz (A) ;
    int64_t plen = A->vdim+1 ; 

    switch (format)
    {
        case GrB_CSR_FORMAT : 
        case GrB_CSC_FORMAT : 
            GB_memcpy (Ap, A->p, plen  * sizeof (GrB_Index), nthreads_max) ;
            GB_memcpy (Ai, A->i, nvals * sizeof (GrB_Index), nthreads_max) ;

        case GrB_DENSE_ROW_FORMAT :
        case GrB_DENSE_COL_FORMAT :
            ASSERT (csc_requested == A->is_csc) ;
            if (A->iso)
            { 
                // expand the iso A->x into the non-iso array Ax
                ASSERT (nvals > 0) ;
                GB_iso_expand (Ax, nvals, A->x, A->type->size, Context) ;
            }
            else
            { 
                GB_memcpy (Ax, A->x, nvals, nthreads_max) ;
            }
            break ;

        default:
        case GrB_COO_FORMAT : 
            GB_OK (GB_extractTuples (Ap, Ai, Ax, &nvals, A->type->code, A,
                Context)) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

