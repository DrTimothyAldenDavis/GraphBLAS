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
{   GB_cov[4432]++ ;
// covered (4432): 13786
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
{   GB_cov[4433]++ ;
// covered (4433): 11059
    GB_WHERE (w, "GrB_Matrix_reduce_BinaryOp (w, M, accum, op, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (op_in) ;
    ASSERT_BINARYOP_OK (op_in, "binary op for reduce-to-vector", GB0) ;

    // check operator types; all must be identical
    if (op_in->xtype != op_in->ztype || op_in->ytype != op_in->ztype)
    {   GB_cov[4434]++ ;
// covered (4434): 2
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
                case GB_INT8_code    : GB_cov[4435]++ ;  monoid = GrB_MIN_MONOID_INT8     ; break ;
// NOT COVERED (4435):
                case GB_INT16_code   : GB_cov[4436]++ ;  monoid = GrB_MIN_MONOID_INT16    ; break ;
// NOT COVERED (4436):
                case GB_INT32_code   : GB_cov[4437]++ ;  monoid = GrB_MIN_MONOID_INT32    ; break ;
// NOT COVERED (4437):
                case GB_INT64_code   : GB_cov[4438]++ ;  monoid = GrB_MIN_MONOID_INT64    ; break ;
// NOT COVERED (4438):
                case GB_UINT8_code   : GB_cov[4439]++ ;  monoid = GrB_MIN_MONOID_UINT8    ; break ;
// NOT COVERED (4439):
                case GB_UINT16_code  : GB_cov[4440]++ ;  monoid = GrB_MIN_MONOID_UINT16   ; break ;
// NOT COVERED (4440):
                case GB_UINT32_code  : GB_cov[4441]++ ;  monoid = GrB_MIN_MONOID_UINT32   ; break ;
// NOT COVERED (4441):
                case GB_UINT64_code  : GB_cov[4442]++ ;  monoid = GrB_MIN_MONOID_UINT64   ; break ;
// NOT COVERED (4442):
                case GB_FP32_code    : GB_cov[4443]++ ;  monoid = GrB_MIN_MONOID_FP32     ; break ;
// NOT COVERED (4443):
                case GB_FP64_code    : GB_cov[4444]++ ;  monoid = GrB_MIN_MONOID_FP64     ; break ;
// NOT COVERED (4444):
                default: ;
            }
            break ;

        case GB_MAX_opcode:

            switch (zcode)
            {
                // 10 MAX monoids: for 10 real types
                case GB_INT8_code    : GB_cov[4445]++ ;  monoid = GrB_MAX_MONOID_INT8     ; break ;
// NOT COVERED (4445):
                case GB_INT16_code   : GB_cov[4446]++ ;  monoid = GrB_MAX_MONOID_INT16    ; break ;
// NOT COVERED (4446):
                case GB_INT32_code   : GB_cov[4447]++ ;  monoid = GrB_MAX_MONOID_INT32    ; break ;
// NOT COVERED (4447):
                case GB_INT64_code   : GB_cov[4448]++ ;  monoid = GrB_MAX_MONOID_INT64    ; break ;
// NOT COVERED (4448):
                case GB_UINT8_code   : GB_cov[4449]++ ;  monoid = GrB_MAX_MONOID_UINT8    ; break ;
// NOT COVERED (4449):
                case GB_UINT16_code  : GB_cov[4450]++ ;  monoid = GrB_MAX_MONOID_UINT16   ; break ;
// NOT COVERED (4450):
                case GB_UINT32_code  : GB_cov[4451]++ ;  monoid = GrB_MAX_MONOID_UINT32   ; break ;
// NOT COVERED (4451):
                case GB_UINT64_code  : GB_cov[4452]++ ;  monoid = GrB_MAX_MONOID_UINT64   ; break ;
// NOT COVERED (4452):
                case GB_FP32_code    : GB_cov[4453]++ ;  monoid = GrB_MAX_MONOID_FP32     ; break ;
// NOT COVERED (4453):
                case GB_FP64_code    : GB_cov[4454]++ ;  monoid = GrB_MAX_MONOID_FP64     ; break ;
// NOT COVERED (4454):
                default: ;
            }
            break ;

        case GB_TIMES_opcode:

            switch (zcode)
            {
                // 12 TIMES monoids: 10 real types, and 2 complex types
                case GB_INT8_code    : GB_cov[4455]++ ;  monoid = GrB_TIMES_MONOID_INT8   ; break ;
// NOT COVERED (4455):
                case GB_INT16_code   : GB_cov[4456]++ ;  monoid = GrB_TIMES_MONOID_INT16  ; break ;
// NOT COVERED (4456):
                case GB_INT32_code   : GB_cov[4457]++ ;  monoid = GrB_TIMES_MONOID_INT32  ; break ;
// NOT COVERED (4457):
                case GB_INT64_code   : GB_cov[4458]++ ;  monoid = GrB_TIMES_MONOID_INT64  ; break ;
// NOT COVERED (4458):
                case GB_UINT8_code   : GB_cov[4459]++ ;  monoid = GrB_TIMES_MONOID_UINT8  ; break ;
// NOT COVERED (4459):
                case GB_UINT16_code  : GB_cov[4460]++ ;  monoid = GrB_TIMES_MONOID_UINT16 ; break ;
// NOT COVERED (4460):
                case GB_UINT32_code  : GB_cov[4461]++ ;  monoid = GrB_TIMES_MONOID_UINT32 ; break ;
// NOT COVERED (4461):
                case GB_UINT64_code  : GB_cov[4462]++ ;  monoid = GrB_TIMES_MONOID_UINT64 ; break ;
// NOT COVERED (4462):
                case GB_FP32_code    : GB_cov[4463]++ ;  monoid = GrB_TIMES_MONOID_FP32   ; break ;
// NOT COVERED (4463):
                case GB_FP64_code    : GB_cov[4464]++ ;  monoid = GrB_TIMES_MONOID_FP64   ; break ;
// NOT COVERED (4464):
                case GB_FC32_code    : GB_cov[4465]++ ;  monoid = GxB_TIMES_FC32_MONOID   ; break ;
// NOT COVERED (4465):
                case GB_FC64_code    : GB_cov[4466]++ ;  monoid = GxB_TIMES_FC64_MONOID   ; break ;
// NOT COVERED (4466):
                default: ;
            }
            break ;

        case GB_PLUS_opcode:

            switch (zcode)
            {
                // 12 PLUS monoids: 10 real types, and 2 complex types
                case GB_INT8_code    : GB_cov[4467]++ ;  monoid = GrB_PLUS_MONOID_INT8    ; break ;
// NOT COVERED (4467):
                case GB_INT16_code   : GB_cov[4468]++ ;  monoid = GrB_PLUS_MONOID_INT16   ; break ;
// NOT COVERED (4468):
                case GB_INT32_code   : GB_cov[4469]++ ;  monoid = GrB_PLUS_MONOID_INT32   ; break ;
// NOT COVERED (4469):
                case GB_INT64_code   : GB_cov[4470]++ ;  monoid = GrB_PLUS_MONOID_INT64   ; break ;
// NOT COVERED (4470):
                case GB_UINT8_code   : GB_cov[4471]++ ;  monoid = GrB_PLUS_MONOID_UINT8   ; break ;
// NOT COVERED (4471):
                case GB_UINT16_code  : GB_cov[4472]++ ;  monoid = GrB_PLUS_MONOID_UINT16  ; break ;
// NOT COVERED (4472):
                case GB_UINT32_code  : GB_cov[4473]++ ;  monoid = GrB_PLUS_MONOID_UINT32  ; break ;
// NOT COVERED (4473):
                case GB_UINT64_code  : GB_cov[4474]++ ;  monoid = GrB_PLUS_MONOID_UINT64  ; break ;
// covered (4474): 5
                case GB_FP32_code    : GB_cov[4475]++ ;  monoid = GrB_PLUS_MONOID_FP32    ; break ;
// NOT COVERED (4475):
                case GB_FP64_code    : GB_cov[4476]++ ;  monoid = GrB_PLUS_MONOID_FP64    ; break ;
// covered (4476): 11046
                case GB_FC32_code    : GB_cov[4477]++ ;  monoid = GxB_PLUS_FC32_MONOID    ; break ;
// NOT COVERED (4477):
                case GB_FC64_code    : GB_cov[4478]++ ;  monoid = GxB_PLUS_FC64_MONOID    ; break ;
// NOT COVERED (4478):
                default: ;
            }
            break ;

        case GB_ANY_opcode:

            switch (zcode)
            {
                // 13 ANY monoids: bool, 10 real types, and 2 complex types
                case GB_BOOL_code    : GB_cov[4479]++ ;  monoid = GxB_ANY_BOOL_MONOID     ; break ;
// NOT COVERED (4479):
                case GB_INT8_code    : GB_cov[4480]++ ;  monoid = GxB_ANY_INT8_MONOID     ; break ;
// NOT COVERED (4480):
                case GB_INT16_code   : GB_cov[4481]++ ;  monoid = GxB_ANY_INT16_MONOID    ; break ;
// NOT COVERED (4481):
                case GB_INT32_code   : GB_cov[4482]++ ;  monoid = GxB_ANY_INT32_MONOID    ; break ;
// NOT COVERED (4482):
                case GB_INT64_code   : GB_cov[4483]++ ;  monoid = GxB_ANY_INT64_MONOID    ; break ;
// NOT COVERED (4483):
                case GB_UINT8_code   : GB_cov[4484]++ ;  monoid = GxB_ANY_UINT8_MONOID    ; break ;
// NOT COVERED (4484):
                case GB_UINT16_code  : GB_cov[4485]++ ;  monoid = GxB_ANY_UINT16_MONOID   ; break ;
// NOT COVERED (4485):
                case GB_UINT32_code  : GB_cov[4486]++ ;  monoid = GxB_ANY_UINT32_MONOID   ; break ;
// NOT COVERED (4486):
                case GB_UINT64_code  : GB_cov[4487]++ ;  monoid = GxB_ANY_UINT64_MONOID   ; break ;
// NOT COVERED (4487):
                case GB_FP32_code    : GB_cov[4488]++ ;  monoid = GxB_ANY_FP32_MONOID     ; break ;
// NOT COVERED (4488):
                case GB_FP64_code    : GB_cov[4489]++ ;  monoid = GxB_ANY_FP64_MONOID     ; break ;
// NOT COVERED (4489):
                case GB_FC32_code    : GB_cov[4490]++ ;  monoid = GxB_ANY_FC32_MONOID     ; break ;
// NOT COVERED (4490):
                case GB_FC64_code    : GB_cov[4491]++ ;  monoid = GxB_ANY_FC64_MONOID     ; break ;
// NOT COVERED (4491):
                default: ;
            }
            break ;

        // 4 boolean monoids: (see also GxB_ANY_BOOL_MONOID above)
        #define B(x) if (zcode == GB_BOOL_code) monoid = x ; break ;
        case GB_LOR_opcode    : GB_cov[4492]++ ;  B (GrB_LOR_MONOID_BOOL)   ;
// NOT COVERED (4492):
        case GB_LAND_opcode   : GB_cov[4493]++ ;  B (GrB_LAND_MONOID_BOOL)  ;
// NOT COVERED (4493):
        case GB_LXOR_opcode   : GB_cov[4494]++ ;  B (GrB_LXOR_MONOID_BOOL)  ;
// NOT COVERED (4494):
        case GB_EQ_opcode     : GB_cov[4495]++ ;  B (GrB_LXNOR_MONOID_BOOL) ;
// NOT COVERED (4495):

        case GB_BOR_opcode:

            switch (zcode)
            {
                // 4 BOR monoids
                case GB_UINT8_code   : GB_cov[4496]++ ;  monoid = GxB_BOR_UINT8_MONOID    ; break ;
// NOT COVERED (4496):
                case GB_UINT16_code  : GB_cov[4497]++ ;  monoid = GxB_BOR_UINT16_MONOID   ; break ;
// NOT COVERED (4497):
                case GB_UINT32_code  : GB_cov[4498]++ ;  monoid = GxB_BOR_UINT32_MONOID   ; break ;
// NOT COVERED (4498):
                case GB_UINT64_code  : GB_cov[4499]++ ;  monoid = GxB_BOR_UINT64_MONOID   ; break ;
// NOT COVERED (4499):
                default: ;
            }
            break ;

        case GB_BAND_opcode:

            switch (zcode)
            {
                // 4 BAND monoids
                case GB_UINT8_code   : GB_cov[4500]++ ;  monoid = GxB_BAND_UINT8_MONOID   ; break ;
// NOT COVERED (4500):
                case GB_UINT16_code  : GB_cov[4501]++ ;  monoid = GxB_BAND_UINT16_MONOID  ; break ;
// NOT COVERED (4501):
                case GB_UINT32_code  : GB_cov[4502]++ ;  monoid = GxB_BAND_UINT32_MONOID  ; break ;
// NOT COVERED (4502):
                case GB_UINT64_code  : GB_cov[4503]++ ;  monoid = GxB_BAND_UINT64_MONOID  ; break ;
// NOT COVERED (4503):
                default: ;
            }
            break ;

        case GB_BXOR_opcode:

            switch (zcode)
            {
                // 4 BXOR monoids
                case GB_UINT8_code   : GB_cov[4504]++ ;  monoid = GxB_BXOR_UINT8_MONOID   ; break ;
// NOT COVERED (4504):
                case GB_UINT16_code  : GB_cov[4505]++ ;  monoid = GxB_BXOR_UINT16_MONOID  ; break ;
// NOT COVERED (4505):
                case GB_UINT32_code  : GB_cov[4506]++ ;  monoid = GxB_BXOR_UINT32_MONOID  ; break ;
// NOT COVERED (4506):
                case GB_UINT64_code  : GB_cov[4507]++ ;  monoid = GxB_BXOR_UINT64_MONOID  ; break ;
// NOT COVERED (4507):
                default: ;
            }
            break ;

        case GB_BXNOR_opcode:

            switch (zcode)
            {
                // 4 BXNOR monoids
                case GB_UINT8_code   : GB_cov[4508]++ ;  monoid = GxB_BXNOR_UINT8_MONOID  ; break ;
// NOT COVERED (4508):
                case GB_UINT16_code  : GB_cov[4509]++ ;  monoid = GxB_BXNOR_UINT16_MONOID ; break ;
// NOT COVERED (4509):
                case GB_UINT32_code  : GB_cov[4510]++ ;  monoid = GxB_BXNOR_UINT32_MONOID ; break ;
// NOT COVERED (4510):
                case GB_UINT64_code  : GB_cov[4511]++ ;  monoid = GxB_BXNOR_UINT64_MONOID ; break ;
// NOT COVERED (4511):
                default: ;
            }
            break ;

        default  : GB_cov[4512]++ ;  ;
// covered (4512): 2
    }

    if (monoid == NULL)
    {   GB_cov[4513]++ ;
// covered (4513): 2
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
{   GB_cov[4514]++ ;
// covered (4514): 85337
    GB_WHERE (w, "GrB_Matrix_reduce_Monoid (w, M, accum, reduce, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;
    GrB_Info info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M,
        accum, monoid, A, desc, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

