//------------------------------------------------------------------------------
// GxB_Context_get_*: get a field in a context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GxB_Context_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_Scalar
(
    GxB_Context Context,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Context_get_Scalar (Context, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int ivalue = 0 ;
    GrB_Info info ;

    switch ((int) field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            dvalue = GB_Context_chunk_get (Context) ;
            break ;

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            ivalue = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_GPU_ID :           // same as GxB_GPU_ID

            ivalue= GB_Context_gpu_id_get (Context) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    switch ((int) field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            info = GB_setElement ((GrB_Matrix) value, NULL, &dvalue, 0, 0,
                GB_FP64_code, Werk) ;
            break ;

        default : 
            info = GB_setElement ((GrB_Matrix) value, NULL, &ivalue, 0, 0,
                GB_INT32_code, Werk) ;
            break ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_String
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_String
(
    GxB_Context Context,
    char * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;  // FIXME set the name of a GrB_Context
}

//------------------------------------------------------------------------------
// GxB_Context_get_ENUM
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_ENUM
(
    GxB_Context Context,
    int * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Context_get_ENUM (Context, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            (*value) = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_GPU_ID :           // same as GxB_GPU_ID

            (*value) = GB_Context_gpu_id_get (Context) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_SIZE
(
    GxB_Context Context,
    size_t * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_VOID
(
    GxB_Context Context,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

