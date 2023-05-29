//------------------------------------------------------------------------------
// GrB_Matrix_set_*: set a field in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Matrix_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_Scalar
(
    GrB_Matrix A,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_set_Scalar (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    float fvalue = 0 ;
    int ivalue = 0 ;
    GrB_Info info ;

    switch ((int) field)
    {

        case GxB_HYPER_SWITCH : 
        case GxB_BITMAP_SWITCH : 

            info = GrB_Scalar_extractElement_FP32 (&fvalue, value) ;
            break ;

        default : 

            info = GrB_Scalar_extractElement_INT32 (&ivalue, value) ;
            break ;
    }

    if (info != GrB_SUCCESS)
    { 
        return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
    } 

    return (GB_matvec_set (A, false, ivalue, fvalue, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_String
(
    GrB_Matrix A,
    char * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;      // TODO: set the name
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_ENUM
(
    GrB_Matrix A,
    int value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_set_ENUM (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_set (A, false, value, 0, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_VOID
(
    GrB_Matrix A,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

