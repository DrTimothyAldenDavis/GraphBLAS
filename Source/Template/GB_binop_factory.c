//------------------------------------------------------------------------------
// GB_binop_factory: switch factory for built-in methods for C=binop(A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The #include'ing file defines the GB_BINOP_WORKER macro, and opcode, xcode,
// ycode, and zcode

{
    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    switch (opcode)
    {

#ifndef GB_BINOP_SUBSET

        //----------------------------------------------------------------------
        case GB_FIRST_opcode   :    // z = x
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _first
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_SECOND_opcode  :    // z = y
        case GB_ANY_opcode  :       // z = y
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _second
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_PAIR_opcode   :    // z = 1
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _pair
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_MIN_opcode     :    // z = min(x,y)
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _min
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_opcode     :    // z = max(x,y)
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _max
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_opcode    :    // z = x + y
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _plus
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_MINUS_opcode   :    // z = x - y
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _minus
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_RMINUS_opcode   :    // z = y - x (reverse minus)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _rminus
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_opcode   :    // z = x * y
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _times
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_DIV_opcode   :      // z = x / y
        //----------------------------------------------------------------------

            // FIRST == DIV for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _div
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_RDIV_opcode   :     // z = y / x (reverse division)
        //----------------------------------------------------------------------

            // SECOND == RDIV for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _rdiv
            #define GB_COMPLEX
            #include "GB_binop_type_factory.c"
            break ;

#ifndef GB_BINOP_SUBSET

        //----------------------------------------------------------------------
        case GB_ISEQ_opcode    :    // z = (x == y)
        //----------------------------------------------------------------------

            // ISEQ == EQ for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _iseq
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_ISNE_opcode    :    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _isne
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_ISGT_opcode    :    // z = (x >  y)
        //----------------------------------------------------------------------

            // ISGT == GT for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _isgt
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_ISLT_opcode    :    // z = (x <  y)
        //----------------------------------------------------------------------

            // ISLT == LT for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _islt
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_ISGE_opcode    :    // z = (x >= y)
        //----------------------------------------------------------------------

            // ISGE == GE for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _isge
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_ISLE_opcode     :    // z = (x <= y)
        //----------------------------------------------------------------------

            // ISLE == LE for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _isle
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_EQ_opcode      :    // z = (x == y)
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _eq
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_NE_opcode      :    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_BINOP_NAME _ne
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_GT_opcode      :    // z = (x >  y)
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _gt
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LT_opcode      :    // z = (x <  y)
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _lt
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_GE_opcode      :    // z = (x >= y)
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _ge
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LE_opcode      :    // z = (x <= y)
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _le
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LOR_opcode     :    // z = x || y
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _lor
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_opcode    :    // z = x && y
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _land
            #include "GB_binop_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_opcode    :    // z = x != y
        //----------------------------------------------------------------------

            #define GB_BINOP_NAME _lxor
            #include "GB_binop_type_factory.c"
            break ;

#if 0

        //----------------------------------------------------------------------
        case GB_ATAN2_opcode    :    // z = atan2 (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _atan2 // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_HYPOT_opcode    :    // z = hypot (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _hypot // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_FMOD_opcode    :    // z = fmod (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _fmod // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_REMAINDER_opcode    :    // z = remainder (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _remainder // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_COPYSIGN_opcode    :    // z = copysign (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _copysign // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_CMPLX_opcode    :    // z = cmplx (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _cmplx // TODO
            break ;

        //----------------------------------------------------------------------
        case GB_POW_opcode    :    // z = pow (x,y)
        //----------------------------------------------------------------------

            // #define GB_BINOP_NAME _cmplx // TODO
            break ;

#endif

#endif

        default: ;
    }
}

