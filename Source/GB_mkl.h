//------------------------------------------------------------------------------
// GB_mkl.h: definitions for using the Intel MKL and/or CBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_MKL_H
#define GB_MKL_H

//==============================================================================
// determine if MKL and/or CBLAS is available
//==============================================================================

#define GB_HAS_MKL_GRAPH 0

#if !defined ( GBCOMPACT )

    #ifdef MKL_ILP64

        // use the Intel MKL ILP64 parallel CBLAS
        #include "mkl.h"
        #define GB_CBLAS_INT MKL_INT
        #define GB_CBLAS_INT_MAX INT64_MAX

        #if ( INTEL_MKL_VERSION >= 20200001 )
            // use the Intel MKL_graph library
            #include "mkl_graph.h"
            #undef  GB_HAS_MKL_GRAPH
            #define GB_HAS_MKL_GRAPH 1
        #endif

    #elif defined ( GB_HAS_CBLAS )

        // FUTURE: other CBLAS packages here
        #include "cblas.h"
        #define GB_CBLAS_INT int
        #define GB_CBLAS_INT_MAX INT32_MAX
        // etc ...

    #endif

#endif

//==============================================================================
// MKL_graph definitions
//==============================================================================

#if GB_HAS_MKL_GRAPH

//------------------------------------------------------------------------------
// GB_info_mkl: map an Intel MKL status to a GraphBLAS GrB_Info
//------------------------------------------------------------------------------

static inline GrB_Info GB_info_mkl      // equivalent GrB_Info
(
    mkl_graph_status_t status           // MKL return status
)
{
    switch (status)
    {
        case MKL_GRAPH_STATUS_SUCCESS         : return (GrB_SUCCESS) ;
        case MKL_GRAPH_STATUS_NOT_INITIALIZED : return (GrB_UNINITIALIZED_OBJECT) ;
        case MKL_GRAPH_STATUS_ALLOC_FAILED    : return (GrB_OUT_OF_MEMORY) ;
        case MKL_GRAPH_STATUS_INVALID_VALUE   : return (GrB_INVALID_VALUE) ;
        case MKL_GRAPH_STATUS_INTERNAL_ERROR  : return (GrB_PANIC) ;
        case MKL_GRAPH_STATUS_NOT_SUPPORTED   : return (GrB_NO_VALUE) ;
        default                               : return (GrB_PANIC) ;
    }
}

//------------------------------------------------------------------------------
// GB_MKL_OK: call an MKL_graph method and check its result
//------------------------------------------------------------------------------

#define GB_MKL_OK(mkl_method)                                               \
{                                                                           \
    info = GB_info_mkl (mkl_method) ;                                       \
    switch (info)                                                           \
    {                                                                       \
        case GrB_SUCCESS :                                                  \
            break ;                                                         \
        case GrB_UNINITIALIZED_OBJECT :                                     \
            GB_MKL_FREE_ALL ;                                               \
            printf ("MKL uninitialized\n") ; \
            return (GB_ERROR (info, (GB_LOG, "MKL_graph uninitialized"))) ; \
        case GrB_OUT_OF_MEMORY :                                            \
            printf ("MKL out of memory\n") ; \
            GB_MKL_FREE_ALL ;                                               \
            return (GB_OUT_OF_MEMORY) ;                                     \
        case GrB_INVALID_VALUE :                                            \
            printf ("MKL invalid\n") ; \
            GB_MKL_FREE_ALL ;                                               \
            return (GB_ERROR (info, (GB_LOG, "MKL_graph invalid value"))) ; \
        case GrB_PANIC :                                                    \
            printf ("MKL panic\n") ; \
            GB_MKL_FREE_ALL ;                                               \
            return (GB_ERROR (info, (GB_LOG, "MKL_graph panic"))) ;         \
        case GrB_NO_VALUE :                                                 \
            printf ("MKL not supported\n") ; \
            GB_MKL_FREE_ALL ;                                               \
            return (GB_ERROR (info, (GB_LOG, "MKL_graph not supported"))) ; \
        default :                                                           \
            GB_MKL_FREE_ALL ;                                               \
            return (GrB_PANIC) ;                                            \
    }                                                                       \
}

//------------------------------------------------------------------------------
// GB_MKL_GRAPH_MATRIX_DESTROY: free an MKL_graph matrix
//------------------------------------------------------------------------------

#define GB_MKL_GRAPH_MATRIX_DESTROY(A_mkl)                  \
{                                                           \
    if (A_mkl != NULL) mkl_graph_matrix_destroy (A_mkl) ;   \
    A_mkl = NULL ;                                          \
}

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_mkl: C=A*B, C<M>=A*B, or C<!M>=A*B using Intel MKL_graph
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3_mkl          // C = A*B using MKL
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_AxB_semiring_mkl: map a GraphBLAS semiring to an Intel MKL semiring
//------------------------------------------------------------------------------

int GB_AxB_semiring_mkl         // return the MKL semiring, or -1 if none.
(
    GB_Opcode add_opcode,       // additive monoid
    GB_Opcode mult_opcode,      // multiply operator
    GB_Opcode xycode            // type of x for z = mult (x,y), except for
                                // z = SECOND(x,y) = y, where xycode is the
                                // type of y
) ;

//------------------------------------------------------------------------------
// GB_type_mkl: map a GraphBLAS type to an Intel MKL type
//------------------------------------------------------------------------------

static inline int GB_type_mkl   // return the MKL type, or -1 if none
(
    GB_Type_code type_code      // GraphBLAS type code
)
{
    switch (type_code)
    {
        case GB_BOOL_code   : return (MKL_GRAPH_TYPE_BOOL) ;
        case GB_INT8_code   : return (-1) ;
        case GB_INT16_code  : return (-1) ;
        case GB_INT32_code  : return (MKL_GRAPH_TYPE_INT32) ;
        case GB_INT64_code  : return (MKL_GRAPH_TYPE_INT64) ;
        case GB_UINT8_code  : return (-1) ;
        case GB_UINT16_code : return (-1) ;
        case GB_UINT32_code : return (-1) ;
        case GB_UINT64_code : return (-1) ;
        case GB_FP32_code   : return (MKL_GRAPH_TYPE_FP32) ;
        case GB_FP64_code   : return (MKL_GRAPH_TYPE_FP64) ;
        case GB_FC32_code   : return (-1) ;
        case GB_FC64_code   : return (-1) ;
        default             : return (-1) ;
    }
}

#endif

//==============================================================================
// CBLAS definitions
//==============================================================================

#if GB_HAS_CBLAS

//------------------------------------------------------------------------------
// GB_cblas_saxpy: Y += alpha*X where X and Y are dense float arrays
//------------------------------------------------------------------------------

void GB_cblas_saxpy         // Y += alpha*X
(
    const int64_t n,        // length of X and Y (note the int64_t type)
    const float alpha,      // scale factor
    const float *X,         // the array X, always stride 1
    float *Y,               // the array Y, always stride 1
    int nthreads            // maximum # of threads to use
) ;

//------------------------------------------------------------------------------
// GB_cblas_daxpy: Y += alpha*X where X and Y are dense double arrays
//------------------------------------------------------------------------------

void GB_cblas_daxpy         // Y += alpha*X
(
    const int64_t n,        // length of X and Y (note the int64_t type)
    const double alpha,     // scale factor
    const double *X,        // the array X, always stride 1
    double *Y,              // the array Y, always stride 1
    int nthreads            // maximum # of threads to use
) ;

#endif
#endif

