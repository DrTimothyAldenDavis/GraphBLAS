//------------------------------------------------------------------------------
// GrB_Vector_build: build a sparse GraphBLAS vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_build.h"

#define GB_VECTOR_BUILD(prefix,type,T)                                        \
GrB_Info GB_EVAL3 (prefix, Vector_build_, T) /* build a vector from tuples*/  \
(                                                                             \
    GrB_Vector w,                   /* vector to build                    */  \
    const GrB_Index *I,             /* array of row indices of tuples     */  \
    const type *X,                  /* array of values of tuples          */  \
    GrB_Index nvals,                /* number of tuples                   */  \
    const GrB_BinaryOp dup          /* binary op to assemble duplicates   */  \
)                                                                             \
{                                                                             \
    GB_WHERE (w, "GrB_Vector_build_" GB_STR(T)                                \
        " (w, I, X, nvals, dup)") ;                                           \
    GB_BURBLE_START ("GrB_Vector_build") ;                                    \
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;                                         \
    ASSERT (GB_VECTOR_OK (w)) ;                                               \
    GrB_Info info = GB_matvec_build ((GrB_Matrix) w, I, NULL, X, nvals, dup,  \
        GB_ ## T ## _code, false, Context) ;                                  \
    ASSERT (GB_IMPLIES (info == GrB_SUCCESS, GB_VECTOR_OK (w))) ;             \
    GB_BURBLE_END ;                                                           \
    return (info) ;                                                           \
}

GB_VECTOR_BUILD (GRB (), bool      , BOOL   )
GB_VECTOR_BUILD (GRB (), int8_t    , INT8   )
GB_VECTOR_BUILD (GRB (), uint8_t   , UINT8  )
GB_VECTOR_BUILD (GRB (), int16_t   , INT16  )
GB_VECTOR_BUILD (GRB (), uint16_t  , UINT16 )
GB_VECTOR_BUILD (GRB (), int32_t   , INT32  )
GB_VECTOR_BUILD (GRB (), uint32_t  , UINT32 )
GB_VECTOR_BUILD (GRB (), int64_t   , INT64  )
GB_VECTOR_BUILD (GRB (), uint64_t  , UINT64 )
GB_VECTOR_BUILD (GRB (), float     , FP32   )
GB_VECTOR_BUILD (GRB (), double    , FP64   )
GB_VECTOR_BUILD (GXB (), GxB_FC32_t, FC32   )
GB_VECTOR_BUILD (GXB (), GxB_FC64_t, FC64   )
GB_VECTOR_BUILD (GRB (), void      , UDT    )

