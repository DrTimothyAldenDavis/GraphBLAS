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
{   GB_cov[11018]++ ;
// covered (11018): 5

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

        case GxB_CONTEXT_CHUNK  : GB_cov[11019]++ ;          // same as GxB_CHUNK
// covered (11019): 2

            dvalue = GB_Context_chunk_get (Context) ;
            break ;

        case GxB_CONTEXT_NTHREADS  : GB_cov[11020]++ ;          // same as GxB_NTHREADS
// covered (11020): 1

            ivalue = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_GPU_ID  : GB_cov[11021]++ ;            // same as GxB_GPU_ID
// covered (11021): 1

            ivalue= GB_Context_gpu_id_get (Context) ;
            break ;

        default  : GB_cov[11022]++ ;  
// covered (11022): 1

            return (GrB_INVALID_VALUE) ;
    }

    switch ((int) field)
    {

        case GxB_CONTEXT_CHUNK  : GB_cov[11023]++ ;          // same as GxB_CHUNK
// covered (11023): 2

            info = GB_setElement ((GrB_Matrix) value, NULL, &dvalue, 0, 0,
                GB_FP64_code, Werk) ;
            break ;

        default  : GB_cov[11024]++ ;  
// covered (11024): 2
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
{   GB_cov[11025]++ ;
// covered (11025): 4

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Context_get_String (Context, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field != GrB_NAME)
    {   GB_cov[11026]++ ;
// NOT COVERED (11026):
        return (GrB_INVALID_VALUE) ;
    }

    (*value) = '\0' ;
    if (Context == GxB_CONTEXT_WORLD)
    {   GB_cov[11027]++ ;
// covered (11027): 2
        // built-in Context
        strcpy (value, "GxB_CONTEXT_WORLD") ;
    }
    else if (Context->user_name_size > 0)
    {   GB_cov[11028]++ ;
// covered (11028): 1
        // user-defined Context, with name defined by GrB_set
        strcpy (value, Context->user_name) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
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
{   GB_cov[11029]++ ;
// covered (11029): 5

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

        case GxB_CONTEXT_NTHREADS  : GB_cov[11030]++ ;          // same as GxB_NTHREADS
// covered (11030): 2

            (*value) = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_GPU_ID  : GB_cov[11031]++ ;            // same as GxB_GPU_ID
// covered (11031): 2

            (*value) = GB_Context_gpu_id_get (Context) ;
            break ;

        default  : GB_cov[11032]++ ;  
// covered (11032): 1

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
{   GB_cov[11033]++ ;
// covered (11033): 3

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Context_get_SIZE (Context, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field != GrB_NAME)
    {   GB_cov[11034]++ ;
// covered (11034): 1
        return (GrB_INVALID_VALUE) ;
    }

    if (Context->user_name != NULL)
    {   GB_cov[11035]++ ;
// covered (11035): 1
        (*value) = Context->user_name_size ;
    }
    else
    {   GB_cov[11036]++ ;
// covered (11036): 1
        (*value) = GxB_MAX_NAME_LEN ;
    }
    return (GrB_SUCCESS) ;
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
{   GB_cov[11037]++ ;
// covered (11037): 1
    return (GrB_NOT_IMPLEMENTED) ;
}

