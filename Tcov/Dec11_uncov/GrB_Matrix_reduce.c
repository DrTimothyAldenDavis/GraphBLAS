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
// monoid.  The result is either assigned to the output scalar c (if accum is
// NULL), or it accumulated in the result c via c = accum(c,t).  If A has no
// entries, the result t is the identity value of the monoid.  Unlike most
// other GraphBLAS operations, this operation uses an accum operator but no
// mask.

#include "GB_reduce.h"
#include "GB_binop.h"

#define GB_MATRIX_TO_SCALAR(prefix,type,T)                                     \
GrB_Info prefix ## Matrix_reduce_ ## T    /* c = accum (c, reduce (A))  */     \
(                                                                              \
    type *c,                        /* result scalar                        */ \
    const GrB_BinaryOp accum,       /* optional accum for c=accum(c,t)      */ \
    const GrB_Monoid monoid,        /* monoid to do the reduction           */ \
    const GrB_Matrix A,             /* matrix to reduce                     */ \
    const GrB_Descriptor desc       /* descriptor (currently unused)        */ \
)                                                                              \
{                                                                              \
    GB_WHERE1 ("Matrix_reduce_" GB_STR(T) " (&c, accum, monoid, A, desc)") ;   \
    GB_BURBLE_START ("GrB_reduce") ;                                           \
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;                                          \
    GrB_Info info = GB_reduce_to_scalar (c, prefix ## T,                       \
        accum, monoid, A, Context) ;                                           \
    GB_BURBLE_END ;                                                            \
    return (info) ;                                                            \
}

GB_MATRIX_TO_SCALAR (GrB_, bool      , BOOL   )
GB_MATRIX_TO_SCALAR (GrB_, int8_t    , INT8   )
GB_MATRIX_TO_SCALAR (GrB_, int16_t   , INT16  )
GB_MATRIX_TO_SCALAR (GrB_, int32_t   , INT32  )
GB_MATRIX_TO_SCALAR (GrB_, int64_t   , INT64  )
GB_MATRIX_TO_SCALAR (GrB_, uint8_t   , UINT8  )
GB_MATRIX_TO_SCALAR (GrB_, uint16_t  , UINT16 )
GB_MATRIX_TO_SCALAR (GrB_, uint32_t  , UINT32 )
GB_MATRIX_TO_SCALAR (GrB_, uint64_t  , UINT64 )
GB_MATRIX_TO_SCALAR (GrB_, float     , FP32   )
GB_MATRIX_TO_SCALAR (GrB_, double    , FP64   )
GB_MATRIX_TO_SCALAR (GxB_, GxB_FC32_t, FC32   )
GB_MATRIX_TO_SCALAR (GxB_, GxB_FC64_t, FC64   )

GrB_Info GrB_Matrix_reduce_UDT      // c = accum (c, reduce_to_scalar (A))
(
    void *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc       // descriptor (currently unused)
)
{   GB_cov[4464]++ ;
// covered (4464): 13786
    // Reduction to a user-defined type requires an assumption about the type
    // of the scalar c.  It's just a void* pointer so its type must be
    // inferred from the other arguments.  The type cannot be found from
    // accum, since accum can be NULL.  The result is computed by the reduce
    // monoid, and no typecasting can be done between user-defined types.
    // Thus, the type of c must be the same as the reduce monoid.

    GB_WHERE1 ("GrB_Matrix_reduce_UDT (&c, accum, monoid, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GrB_Info info = GB_reduce_to_scalar (c, monoid->op->ztype,
        accum, monoid, A, Context) ;
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
    const GrB_BinaryOp op_in,       // reduce operator for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{   GB_cov[4465]++ ;
// covered (4465): 11059
    GB_WHERE (w, "GrB_Matrix_reduce_BinaryOp (w, M, accum, op, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (op_in) ;
    ASSERT_BINARYOP_OK (op_in, "binary op for reduce-to-vector", GB0) ;

    // check operator types; all must be identical
    if (op_in->xtype != op_in->ztype || op_in->ytype != op_in->ztype)
    {   GB_cov[4466]++ ;
// covered (4466): 2
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Invalid binary operator:"
            " z=%s(x,y) has no equivalent monoid\n", op_in->name) ;
    }

    //--------------------------------------------------------------------------
    // convert the binary op_in to its corresponding monoid
    //--------------------------------------------------------------------------

    GrB_Monoid monoid = NULL ;
    GrB_BinaryOp op = GB_boolean_rename_op (op_in) ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Opcode opcode = op->opcode ;

    switch (opcode)
    {

        case GB_MIN_opcode:

            switch (zcode)
            {
                // 10 MIN monoids: for 10 real types
                case GB_INT8_code    : GB_cov[4467]++ ;  monoid = GrB_MIN_MONOID_INT8     ; break ;
// NOT COVERED (4467):
                case GB_INT16_code   : GB_cov[4468]++ ;  monoid = GrB_MIN_MONOID_INT16    ; break ;
// NOT COVERED (4468):
                case GB_INT32_code   : GB_cov[4469]++ ;  monoid = GrB_MIN_MONOID_INT32    ; break ;
// NOT COVERED (4469):
                case GB_INT64_code   : GB_cov[4470]++ ;  monoid = GrB_MIN_MONOID_INT64    ; break ;
// NOT COVERED (4470):
                case GB_UINT8_code   : GB_cov[4471]++ ;  monoid = GrB_MIN_MONOID_UINT8    ; break ;
// NOT COVERED (4471):
                case GB_UINT16_code  : GB_cov[4472]++ ;  monoid = GrB_MIN_MONOID_UINT16   ; break ;
// NOT COVERED (4472):
                case GB_UINT32_code  : GB_cov[4473]++ ;  monoid = GrB_MIN_MONOID_UINT32   ; break ;
// NOT COVERED (4473):
                case GB_UINT64_code  : GB_cov[4474]++ ;  monoid = GrB_MIN_MONOID_UINT64   ; break ;
// NOT COVERED (4474):
                case GB_FP32_code    : GB_cov[4475]++ ;  monoid = GrB_MIN_MONOID_FP32     ; break ;
// NOT COVERED (4475):
                case GB_FP64_code    : GB_cov[4476]++ ;  monoid = GrB_MIN_MONOID_FP64     ; break ;
// NOT COVERED (4476):
                default: ;
            }
            break ;

        case GB_MAX_opcode:

            switch (zcode)
            {
                // 10 MAX monoids: for 10 real types
                case GB_INT8_code    : GB_cov[4477]++ ;  monoid = GrB_MAX_MONOID_INT8     ; break ;
// NOT COVERED (4477):
                case GB_INT16_code   : GB_cov[4478]++ ;  monoid = GrB_MAX_MONOID_INT16    ; break ;
// NOT COVERED (4478):
                case GB_INT32_code   : GB_cov[4479]++ ;  monoid = GrB_MAX_MONOID_INT32    ; break ;
// NOT COVERED (4479):
                case GB_INT64_code   : GB_cov[4480]++ ;  monoid = GrB_MAX_MONOID_INT64    ; break ;
// NOT COVERED (4480):
                case GB_UINT8_code   : GB_cov[4481]++ ;  monoid = GrB_MAX_MONOID_UINT8    ; break ;
// NOT COVERED (4481):
                case GB_UINT16_code  : GB_cov[4482]++ ;  monoid = GrB_MAX_MONOID_UINT16   ; break ;
// NOT COVERED (4482):
                case GB_UINT32_code  : GB_cov[4483]++ ;  monoid = GrB_MAX_MONOID_UINT32   ; break ;
// NOT COVERED (4483):
                case GB_UINT64_code  : GB_cov[4484]++ ;  monoid = GrB_MAX_MONOID_UINT64   ; break ;
// NOT COVERED (4484):
                case GB_FP32_code    : GB_cov[4485]++ ;  monoid = GrB_MAX_MONOID_FP32     ; break ;
// NOT COVERED (4485):
                case GB_FP64_code    : GB_cov[4486]++ ;  monoid = GrB_MAX_MONOID_FP64     ; break ;
// NOT COVERED (4486):
                default: ;
            }
            break ;

        case GB_TIMES_opcode:

            switch (zcode)
            {
                // 12 TIMES monoids: 10 real types, and 2 complex types
                case GB_INT8_code    : GB_cov[4487]++ ;  monoid = GrB_TIMES_MONOID_INT8   ; break ;
// NOT COVERED (4487):
                case GB_INT16_code   : GB_cov[4488]++ ;  monoid = GrB_TIMES_MONOID_INT16  ; break ;
// NOT COVERED (4488):
                case GB_INT32_code   : GB_cov[4489]++ ;  monoid = GrB_TIMES_MONOID_INT32  ; break ;
// NOT COVERED (4489):
                case GB_INT64_code   : GB_cov[4490]++ ;  monoid = GrB_TIMES_MONOID_INT64  ; break ;
// NOT COVERED (4490):
                case GB_UINT8_code   : GB_cov[4491]++ ;  monoid = GrB_TIMES_MONOID_UINT8  ; break ;
// NOT COVERED (4491):
                case GB_UINT16_code  : GB_cov[4492]++ ;  monoid = GrB_TIMES_MONOID_UINT16 ; break ;
// NOT COVERED (4492):
                case GB_UINT32_code  : GB_cov[4493]++ ;  monoid = GrB_TIMES_MONOID_UINT32 ; break ;
// NOT COVERED (4493):
                case GB_UINT64_code  : GB_cov[4494]++ ;  monoid = GrB_TIMES_MONOID_UINT64 ; break ;
// NOT COVERED (4494):
                case GB_FP32_code    : GB_cov[4495]++ ;  monoid = GrB_TIMES_MONOID_FP32   ; break ;
// NOT COVERED (4495):
                case GB_FP64_code    : GB_cov[4496]++ ;  monoid = GrB_TIMES_MONOID_FP64   ; break ;
// NOT COVERED (4496):
                case GB_FC32_code    : GB_cov[4497]++ ;  monoid = GxB_TIMES_FC32_MONOID   ; break ;
// NOT COVERED (4497):
                case GB_FC64_code    : GB_cov[4498]++ ;  monoid = GxB_TIMES_FC64_MONOID   ; break ;
// NOT COVERED (4498):
                default: ;
            }
            break ;

        case GB_PLUS_opcode:

            switch (zcode)
            {
                // 12 PLUS monoids: 10 real types, and 2 complex types
                case GB_INT8_code    : GB_cov[4499]++ ;  monoid = GrB_PLUS_MONOID_INT8    ; break ;
// NOT COVERED (4499):
                case GB_INT16_code   : GB_cov[4500]++ ;  monoid = GrB_PLUS_MONOID_INT16   ; break ;
// NOT COVERED (4500):
                case GB_INT32_code   : GB_cov[4501]++ ;  monoid = GrB_PLUS_MONOID_INT32   ; break ;
// NOT COVERED (4501):
                case GB_INT64_code   : GB_cov[4502]++ ;  monoid = GrB_PLUS_MONOID_INT64   ; break ;
// NOT COVERED (4502):
                case GB_UINT8_code   : GB_cov[4503]++ ;  monoid = GrB_PLUS_MONOID_UINT8   ; break ;
// NOT COVERED (4503):
                case GB_UINT16_code  : GB_cov[4504]++ ;  monoid = GrB_PLUS_MONOID_UINT16  ; break ;
// NOT COVERED (4504):
                case GB_UINT32_code  : GB_cov[4505]++ ;  monoid = GrB_PLUS_MONOID_UINT32  ; break ;
// NOT COVERED (4505):
                case GB_UINT64_code  : GB_cov[4506]++ ;  monoid = GrB_PLUS_MONOID_UINT64  ; break ;
// covered (4506): 5
                case GB_FP32_code    : GB_cov[4507]++ ;  monoid = GrB_PLUS_MONOID_FP32    ; break ;
// NOT COVERED (4507):
                case GB_FP64_code    : GB_cov[4508]++ ;  monoid = GrB_PLUS_MONOID_FP64    ; break ;
// covered (4508): 11046
                case GB_FC32_code    : GB_cov[4509]++ ;  monoid = GxB_PLUS_FC32_MONOID    ; break ;
// NOT COVERED (4509):
                case GB_FC64_code    : GB_cov[4510]++ ;  monoid = GxB_PLUS_FC64_MONOID    ; break ;
// NOT COVERED (4510):
                default: ;
            }
            break ;

        case GB_ANY_opcode:

            switch (zcode)
            {
                // 13 ANY monoids: bool, 10 real types, and 2 complex types
                case GB_BOOL_code    : GB_cov[4511]++ ;  monoid = GxB_ANY_BOOL_MONOID     ; break ;
// NOT COVERED (4511):
                case GB_INT8_code    : GB_cov[4512]++ ;  monoid = GxB_ANY_INT8_MONOID     ; break ;
// NOT COVERED (4512):
                case GB_INT16_code   : GB_cov[4513]++ ;  monoid = GxB_ANY_INT16_MONOID    ; break ;
// NOT COVERED (4513):
                case GB_INT32_code   : GB_cov[4514]++ ;  monoid = GxB_ANY_INT32_MONOID    ; break ;
// NOT COVERED (4514):
                case GB_INT64_code   : GB_cov[4515]++ ;  monoid = GxB_ANY_INT64_MONOID    ; break ;
// NOT COVERED (4515):
                case GB_UINT8_code   : GB_cov[4516]++ ;  monoid = GxB_ANY_UINT8_MONOID    ; break ;
// NOT COVERED (4516):
                case GB_UINT16_code  : GB_cov[4517]++ ;  monoid = GxB_ANY_UINT16_MONOID   ; break ;
// NOT COVERED (4517):
                case GB_UINT32_code  : GB_cov[4518]++ ;  monoid = GxB_ANY_UINT32_MONOID   ; break ;
// NOT COVERED (4518):
                case GB_UINT64_code  : GB_cov[4519]++ ;  monoid = GxB_ANY_UINT64_MONOID   ; break ;
// NOT COVERED (4519):
                case GB_FP32_code    : GB_cov[4520]++ ;  monoid = GxB_ANY_FP32_MONOID     ; break ;
// NOT COVERED (4520):
                case GB_FP64_code    : GB_cov[4521]++ ;  monoid = GxB_ANY_FP64_MONOID     ; break ;
// NOT COVERED (4521):
                case GB_FC32_code    : GB_cov[4522]++ ;  monoid = GxB_ANY_FC32_MONOID     ; break ;
// NOT COVERED (4522):
                case GB_FC64_code    : GB_cov[4523]++ ;  monoid = GxB_ANY_FC64_MONOID     ; break ;
// NOT COVERED (4523):
                default: ;
            }
            break ;

        // 4 boolean monoids: (see also GxB_ANY_BOOL_MONOID above)
        #define B(x) if (zcode == GB_BOOL_code) monoid = x ; break ;
        case GB_LOR_opcode    : GB_cov[4524]++ ;  B (GrB_LOR_MONOID_BOOL)   ;
// NOT COVERED (4524):
        case GB_LAND_opcode   : GB_cov[4525]++ ;  B (GrB_LAND_MONOID_BOOL)  ;
// NOT COVERED (4525):
        case GB_LXOR_opcode   : GB_cov[4526]++ ;  B (GrB_LXOR_MONOID_BOOL)  ;
// NOT COVERED (4526):
        case GB_EQ_opcode     : GB_cov[4527]++ ;  B (GrB_LXNOR_MONOID_BOOL) ;
// NOT COVERED (4527):

        case GB_BOR_opcode:

            switch (zcode)
            {
                // 4 BOR monoids
                case GB_UINT8_code   : GB_cov[4528]++ ;  monoid = GxB_BOR_UINT8_MONOID    ; break ;
// NOT COVERED (4528):
                case GB_UINT16_code  : GB_cov[4529]++ ;  monoid = GxB_BOR_UINT16_MONOID   ; break ;
// NOT COVERED (4529):
                case GB_UINT32_code  : GB_cov[4530]++ ;  monoid = GxB_BOR_UINT32_MONOID   ; break ;
// NOT COVERED (4530):
                case GB_UINT64_code  : GB_cov[4531]++ ;  monoid = GxB_BOR_UINT64_MONOID   ; break ;
// NOT COVERED (4531):
                default: ;
            }
            break ;

        case GB_BAND_opcode:

            switch (zcode)
            {
                // 4 BAND monoids
                case GB_UINT8_code   : GB_cov[4532]++ ;  monoid = GxB_BAND_UINT8_MONOID   ; break ;
// NOT COVERED (4532):
                case GB_UINT16_code  : GB_cov[4533]++ ;  monoid = GxB_BAND_UINT16_MONOID  ; break ;
// NOT COVERED (4533):
                case GB_UINT32_code  : GB_cov[4534]++ ;  monoid = GxB_BAND_UINT32_MONOID  ; break ;
// NOT COVERED (4534):
                case GB_UINT64_code  : GB_cov[4535]++ ;  monoid = GxB_BAND_UINT64_MONOID  ; break ;
// NOT COVERED (4535):
                default: ;
            }
            break ;

        case GB_BXOR_opcode:

            switch (zcode)
            {
                // 4 BXOR monoids
                case GB_UINT8_code   : GB_cov[4536]++ ;  monoid = GxB_BXOR_UINT8_MONOID   ; break ;
// NOT COVERED (4536):
                case GB_UINT16_code  : GB_cov[4537]++ ;  monoid = GxB_BXOR_UINT16_MONOID  ; break ;
// NOT COVERED (4537):
                case GB_UINT32_code  : GB_cov[4538]++ ;  monoid = GxB_BXOR_UINT32_MONOID  ; break ;
// NOT COVERED (4538):
                case GB_UINT64_code  : GB_cov[4539]++ ;  monoid = GxB_BXOR_UINT64_MONOID  ; break ;
// NOT COVERED (4539):
                default: ;
            }
            break ;

        case GB_BXNOR_opcode:

            switch (zcode)
            {
                // 4 BXNOR monoids
                case GB_UINT8_code   : GB_cov[4540]++ ;  monoid = GxB_BXNOR_UINT8_MONOID  ; break ;
// NOT COVERED (4540):
                case GB_UINT16_code  : GB_cov[4541]++ ;  monoid = GxB_BXNOR_UINT16_MONOID ; break ;
// NOT COVERED (4541):
                case GB_UINT32_code  : GB_cov[4542]++ ;  monoid = GxB_BXNOR_UINT32_MONOID ; break ;
// NOT COVERED (4542):
                case GB_UINT64_code  : GB_cov[4543]++ ;  monoid = GxB_BXNOR_UINT64_MONOID ; break ;
// NOT COVERED (4543):
                default: ;
            }
            break ;

        default  : GB_cov[4544]++ ;  ;
// covered (4544): 2
    }

    if (monoid == NULL)
    {   GB_cov[4545]++ ;
// covered (4545): 2
        // op_in binary operator does not correspond to a known monoid
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Invalid binary operator:"
            " z=%s(x,y) has no equivalent monoid\n", op_in->name) ;
    }

    ASSERT_MONOID_OK (monoid, "monoid for reduce-to-vector", GB0) ;

    GrB_Info info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M,
        accum, monoid, A, desc, Context) ;
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
    const GrB_Monoid monoid,        // reduce monoid for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{   GB_cov[4546]++ ;
// covered (4546): 85337
    GB_WHERE (w, "GrB_Matrix_reduce_Monoid (w, M, accum, reduce, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GrB_Info info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M,
        accum, monoid, A, desc, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

