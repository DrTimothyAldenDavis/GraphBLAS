//------------------------------------------------------------------------------
// GB_matvec_enum_get: get an enum field from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_enum_get (GrB_Matrix A, int32_t *value, int field)
{
    switch (field)
    {
        case GrB_STORAGE_ORIENTATION_HINT  : GB_cov[4362]++ ;  
// covered (4362): 7

            (*value) = (A->is_csc) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
            break ;

        case GrB_EL_TYPE_CODE  : GB_cov[4363]++ ;  
// covered (4363): 6

            (*value) = GB_type_code_get (A->type->code) ;
            break ;

        case GxB_SPARSITY_CONTROL  : GB_cov[4364]++ ;  
// covered (4364): 3

            (*value) = A->sparsity_control ;
            break ;

        case GxB_SPARSITY_STATUS  : GB_cov[4365]++ ;  
// covered (4365): 6

            (*value) = GB_sparsity (A) ;
            break ;

        case GxB_HYPER_HASH  : GB_cov[4366]++ ;  
// NOT COVERED (4366):

            (*value) = !(A->no_hyper_hash) ;
            break ;

        case GxB_FORMAT  : GB_cov[4367]++ ;  
// covered (4367): 6

            (*value) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            break ;

        default  : GB_cov[4368]++ ;  
// covered (4368): 11
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

