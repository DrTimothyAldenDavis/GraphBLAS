//------------------------------------------------------------------------------
// GB_matvec_name_size_get: get max size of a matrix/vector/scalar name or type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_name_size_get (GrB_Matrix A, size_t *value, int field)
{
    const char *name = NULL ;

    switch (field)
    {

        case GrB_NAME  : GB_cov[6409]++ ;      
// covered (6409): 5
            name = A->user_name ;
            break ;

        case GxB_JIT_C_NAME  : GB_cov[6410]++ ;  
// NOT COVERED (6410):
            name = A->type->name ;
            break ;

        case GrB_ELTYPE_STRING  : GB_cov[6411]++ ;  
// covered (6411): 3
            name = GB_type_name_get (A->type) ;
            break ;

        default  : GB_cov[6412]++ ;  
// covered (6412): 3
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (name == NULL) ? 1 : (strlen (name) + 1) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

