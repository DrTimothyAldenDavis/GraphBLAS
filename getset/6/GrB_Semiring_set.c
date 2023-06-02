//------------------------------------------------------------------------------
// GrB_Semiring_set_*: set a field in a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Semiring_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_Scalar
(
    GrB_Semiring semiring,
    GrB_Scalar value,
    GrB_Field field
)
{   GB_cov[10832]++ ;
// covered (10832): 1
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_String
(
    GrB_Semiring semiring,
    char * value,
    GrB_Field field
)
{   GB_cov[10833]++ ;
// covered (10833): 1

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_set_String (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    if (semiring->header_size == 0 || field != GrB_NAME)
    {   GB_cov[10834]++ ;
// NOT COVERED (10834):
        // built-in semirings may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_user_name_set (&(semiring->user_name),
        &(semiring->user_name_size), value)) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_ENUM
(
    GrB_Semiring semiring,
    int value,
    GrB_Field field
)
{   GB_cov[10835]++ ;
// covered (10835): 1
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_VOID
(
    GrB_Semiring semiring,
    void * value,
    GrB_Field field,
    size_t size
)
{   GB_cov[10836]++ ;
// covered (10836): 1
    return (GrB_NOT_IMPLEMENTED) ;
}

