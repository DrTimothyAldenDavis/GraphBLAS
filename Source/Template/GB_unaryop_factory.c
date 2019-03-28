//------------------------------------------------------------------------------
// GB_unaryop_factory.c:  switch factory for unary operators and 2 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Switch factory for applying a unary operator, where the input and output
// types differ (the worker does the typecasting as well).  This file is
// #include'd into GB_apply_op.c and GB_transpose_op.c, which must define
// the GrB_UnaryOp op and the GrB_Type atype.

{

    // switch factory for two types, controlled by code1 and code2
    GB_Type_code code1 = op->ztype->code ;      // defines ztype
    GB_Type_code code2 = atype->code ;          // defines atype

    ASSERT (code1 <= GB_UDT_code) ;
    ASSERT (code2 <= GB_UDT_code) ;

    // GB_BOP(x) is for boolean x, GB_IOP(x) for signed integer (int*),
    // GB_UOP(x) is for unsigned integer (uint*), and GB_FOP(x) is for float,
    // and GB_DOP(x) is for double.

    // NOTE: some of these operators z=f(x) do not depend on x, like z=1.  x is
    // read anyway, but the compiler can remove that as dead code if it is able
    // to.  gcc with -Wunused-but-set-variable will complain, but there's no
    // simple way to silence this spurious warning.  Ignore it.

    switch (op->opcode)
    {

        case GB_ONE_opcode :       // z = 1

            #define GB_BOP(x) true
            #define GB_IOP(x) 1
            #define GB_UOP(x) 1
            #define GB_FOP(x) 1
            #define GB_DOP(x) 1
            #include "GB_2type_factory.c"
            break ;

        case GB_IDENTITY_opcode :  // z = x

            #define GB_BOP(x) x
            #define GB_IOP(x) x
            #define GB_UOP(x) x
            #define GB_FOP(x) x
            #define GB_DOP(x) x
            #include "GB_2type_factory.c"
            break ;

        case GB_AINV_opcode :      // z = -x

            #define GB_BOP(x)  x
            #define GB_IOP(x) -x
            #define GB_UOP(x) -x
            #define GB_FOP(x) -x
            #define GB_DOP(x) -x
            #include "GB_2type_factory.c"
            break ;

        case GB_ABS_opcode :       // z = abs(x)

            #define GB_BOP(x) x
            #define GB_IOP(x) GB_IABS(x)
            #define GB_UOP(x) x
            #define GB_FOP(x) fabsf(x)
            #define GB_DOP(x) fabs(x)
            #include "GB_2type_factory.c"
            break ;

        case GB_MINV_opcode :      // z = 1/x

            // see Source/GB.h discussion on boolean and integer division
            #define GB_BOP(x) true
            #define GB_IOP(x) GB_IMINV_SIGNED(x,GB_BITS)
            #define GB_UOP(x) GB_IMINV_UNSIGNED(x,GB_BITS)
            #define GB_FOP(x) 1./x
            #define GB_DOP(x) 1./x
            #include "GB_2type_factory.c"
            break ;

        case GB_LNOT_opcode :      // z = ! (x != 0)

            #define GB_BOP(x) !x
            #define GB_IOP(x) (!(x != 0))
            #define GB_UOP(x) (!(x != 0))
            #define GB_FOP(x) (!(x != 0))
            #define GB_DOP(x) (!(x != 0))
            #include "GB_2type_factory.c"
            break ;

        default: ;
    }
}
