//------------------------------------------------------------------------------
// GB_matvec_name_get: get a name of a matrix or its type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_name_get (GrB_Matrix A, char *name, int field)
{
    const char *typename ;
    (*name) = '\0' ;

    switch (field)
    {

        case GrB_NAME  : GB_cov[6403]++ ;  
// covered (6403): 6
            if (A->user_name_size > 0)
            {
                strcpy (name, A->user_name) ;
            }
            break ;

        case GxB_JIT_C_NAME  : GB_cov[6404]++ ;  
// NOT COVERED (6404):
            strcpy (name, A->type->name) ;
            break ;

        case GrB_ELTYPE_STRING  : GB_cov[6405]++ ;  
// covered (6405): 3
            typename = GB_type_name_get (A->type) ;
            if (typename != NULL)
            {
                strcpy (name, typename) ;
            }
            break ;

        default  : GB_cov[6406]++ ;  
// covered (6406): 2
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

