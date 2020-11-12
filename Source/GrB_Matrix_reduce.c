//------------------------------------------------------------------------------
// GrB_Matrix_reduce: reduce a matrix to a vector or scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_TYPE: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// Reduce entries in a matrix to a scalar, c = accum (c, reduce_to_scalar(A)))

// All entries in the matrix are "summed" to a single scalar t using the reduce
// monoid, which must be associative (otherwise the results are undefined).
// The result is either assigned to the output scalar c (if accum is NULL), or
// it accumulated in the result c via c = accum(c,t).  If A has no entries, the
// result t is the identity value of the monoid.  Unlike most other GraphBLAS
// operations, this operation uses an accum operator but no mask.

#include "GB_reduce.h"

#define GB_MATRIX_TO_SCALAR(prefix,type,T)                                     \
GrB_Info prefix ## Matrix_reduce_ ## T    /* c = accum (c, reduce (A))  */     \
(                                                                              \
    type *c,                        /* result scalar                        */ \
    const GrB_BinaryOp accum,       /* optional accum for c=accum(c,t)      */ \
    const GrB_Monoid reduce,        /* monoid to do the reduction           */ \
    const GrB_Matrix A,             /* matrix to reduce                     */ \
    const GrB_Descriptor desc       /* descriptor (currently unused)        */ \
)                                                                              \
{                                                                              \
    GB_WHERE1 ("Matrix_reduce_" GB_STR(T) " (&c, accum, reduce, A, desc)") ;   \
    GB_BURBLE_START ("GrB_reduce") ;                                           \
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;                                          \
    GrB_Info info = GB_reduce_to_scalar (c, prefix ## T, accum, reduce, A,     \
        Context) ;                                                             \
    GB_BURBLE_END ;                                                            \
    return (info) ;                                                            \
}

GB_MATRIX_TO_SCALAR (GrB_, bool      , BOOL   )
GB_MATRIX_TO_SCALAR (GrB_, int8_t    , INT8   )
GB_MATRIX_TO_SCALAR (GrB_, uint8_t   , UINT8  )
GB_MATRIX_TO_SCALAR (GrB_, int16_t   , INT16  )
GB_MATRIX_TO_SCALAR (GrB_, uint16_t  , UINT16 )
GB_MATRIX_TO_SCALAR (GrB_, int32_t   , INT32  )
GB_MATRIX_TO_SCALAR (GrB_, uint32_t  , UINT32 )
GB_MATRIX_TO_SCALAR (GrB_, int64_t   , INT64  )
GB_MATRIX_TO_SCALAR (GrB_, uint64_t  , UINT64 )
GB_MATRIX_TO_SCALAR (GrB_, float     , FP32   )
GB_MATRIX_TO_SCALAR (GrB_, double    , FP64   )
GB_MATRIX_TO_SCALAR (GxB_, GxB_FC32_t, FC32   )
GB_MATRIX_TO_SCALAR (GxB_, GxB_FC64_t, FC64   )

GrB_Info GrB_Matrix_reduce_UDT      // c = accum (c, reduce_to_scalar (A))
(
    void *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid reduce,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc       // descriptor (currently unused)
)
{ 
    // Reduction to a user-defined type requires an assumption about the type
    // of the scalar c.  It's just a void* pointer so its type must be
    // inferred from the other arguments.  The type cannot be found from
    // accum, since accum can be NULL.  The result is computed by the reduce
    // monoid, and no typecasting can be done between user-defined types.
    // Thus, the type of c must be the same as the reduce monoid.

    GB_WHERE1 ("GrB_Matrix_reduce_UDT (&c, accum, reduce, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (reduce) ;
    GrB_Info info = GB_reduce_to_scalar (c, reduce->op->ztype, accum, reduce,
        A, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_BinaryOp: reduce a matrix to a vector via a binary op
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_reduce_BinaryOp
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp reduce_op,   // reduce operator for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{ 
    GB_WHERE (w, "GrB_Matrix_reduce_BinaryOp (w, M, accum, reduce, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GB_RETURN_IF_NULL_OR_FAULTY (reduce_op) ;
    ASSERT_BINARYOP_OK (reduce_op, "op for reduce-to-vector", GB0) ;
    GrB_Info info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M,
        accum, reduce_op, NULL, A, desc, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_Monoid: reduce a matrix to a vector via a monoid
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_reduce_Monoid   // w<M> = accum (w,reduce(A))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid reduce_monoid, // reduce monoid for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{ 
    GB_WHERE (w, "GrB_Matrix_reduce_Monoid (w, M, accum, reduce, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GB_RETURN_IF_NULL_OR_FAULTY (reduce_monoid) ;
    ASSERT_MONOID_OK (reduce_monoid, "monoid for reduce-to-vector", GB0) ;
    GrB_Info info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M,
        accum, NULL, reduce_monoid, A, desc, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

