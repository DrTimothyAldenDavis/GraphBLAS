//------------------------------------------------------------------------------
// GB_matvec_set: set a field in a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include "GB_transpose.h"
#define GB_FREE_ALL ;

GrB_Info GB_matvec_set
(
    GrB_Matrix A,
    bool is_vector,         // true if A is a GrB_Vector
    int ivalue,
    double dvalue,
    int field,
    GB_Werk Werk
)
{

    GrB_Info info ;
    GB_BURBLE_START ("GrB_set") ;

    int format = ivalue ;

    switch (field)
    {

        case GxB_HYPER_SWITCH  : GB_cov[4379]++ ;  
// covered (4379): 3

            if (is_vector)
            {   GB_cov[4380]++ ;
// covered (4380): 1
                return (GrB_INVALID_VALUE) ;
            }
            A->hyper_switch = (float) dvalue ;
            break ;

        case GxB_HYPER_HASH  : GB_cov[4381]++ ;  
// NOT COVERED (4381):

            A->no_hyper_hash = !((bool) ivalue) ;
            break ;

        case GxB_BITMAP_SWITCH  : GB_cov[4382]++ ;  
// covered (4382): 2

            A->bitmap_switch = (float) dvalue ;
            break ;

        case GxB_SPARSITY_CONTROL  : GB_cov[4383]++ ;  
// covered (4383): 5

            A->sparsity_control = GB_sparsity_control (ivalue, (int64_t) (-1)) ;
            break ;

        case GrB_STORAGE_ORIENTATION_HINT  : GB_cov[4384]++ ;  
// covered (4384): 4

            format = (ivalue == GrB_COLMAJOR) ? GxB_BY_COL : GxB_BY_ROW ;
            // fall through to the GxB_FORMAT case

        case GxB_FORMAT  : GB_cov[4385]++ ;  
// covered (4385): 5

            if (is_vector)
            {   GB_cov[4386]++ ;
// covered (4386): 1
                // the hint is ignored
                return (GrB_SUCCESS) ;
            }
            if (! (format == GxB_BY_ROW || format == GxB_BY_COL))
            {   GB_cov[4387]++ ;
// covered (4387): 1
                return (GrB_INVALID_VALUE) ;
            }
            bool new_csc = (format != GxB_BY_ROW) ;
            // conform the matrix to the new by-row/by-col format
            if (A->is_csc != new_csc)
            {   GB_cov[4388]++ ;
// covered (4388): 3
                // A = A', done in-place, and change to the new format.
                GB_BURBLE_N (GB_nnz (A), "(transpose) ") ;
                GB_OK (GB_transpose_in_place (A, new_csc, Werk)) ;
                ASSERT (A->is_csc == new_csc) ;
                ASSERT (GB_JUMBLED_OK (A)) ;
            }
            break ;

        default  : GB_cov[4389]++ ;  
// covered (4389): 1
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // conform the matrix to its new desired sparsity structure
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A set before conform", GB0) ;
    GB_OK (GB_conform (A, Werk)) ;
    GB_BURBLE_END ;
    ASSERT_MATRIX_OK (A, "A set after conform", GB0) ;
    return (GrB_SUCCESS) ;
}

