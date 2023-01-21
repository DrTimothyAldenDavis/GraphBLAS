//------------------------------------------------------------------------------
// GB_cuda_upscale_identity: return a 32-bit representation of the identity
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.h"
extern "C"
{
    #include "GB_binop.h"
}

void GB_cuda_upscale_identity
(
    GB_void *identity_upscaled,     // output: at least sizeof (int32_t)
    GrB_Monoid monoid               // input: monoid to upscale
)
{

    //--------------------------------------------------------------------------
    // get the monoid and initialize its upscaled identity value
    //--------------------------------------------------------------------------

    GrB_BinaryOp op = GB_boolean_rename_op (monoid->op) ;
    size_t zsize = op->ztype->size ;

    memset (identity_upscaled, 0, GB_IMAX (zsize, sizeof (int32_t))) ;
    mempcy (identity_upscaled, monoid->identity, zsize) ;

    if (zsize >= sizeof (int32_t))
    {
        // no more work to do
        return ;
    }

    //--------------------------------------------------------------------------
    // upscale the identity value
    //--------------------------------------------------------------------------

    GB_Type_code zcode = op->ztype->code ;
    GB_Opcode opcode = op->opcode ;

    #define SET(type,id)                                        \
    {                                                           \
        type id32 = (type) (id) ;                               \
        memcpy (identity_upscaled, id32, sizeof (int32_t)) ;    \
        return ;                                                \
    }

    switch (opcode)
    {

        case GB_MIN_binop_code:

            switch (zcode)
            {
                case GB_INT8_code   : SET (int32_t,  INT8_MAX) ;
                case GB_INT16_code  : SET (int32_t,  INT16_MAX) ;
                case GB_UINT8_code  : SET (uint32_t, UINT8_MAX) ;
                case GB_UINT16_code : SET (uint32_t, UINT16_MAX) ;
                default: ;
            }
            break ;

        case GB_MAX_binop_code:

            switch (zcode)
            {
                case GB_INT8_code   : SET (int32_t,  INT8_MIN) ;
                case GB_INT16_code  : SET (int32_t,  INT16_MIN) ;
                case GB_UINT8_code  : SET (uint32_t, 0) ;
                case GB_UINT16_code : SET (uint32_t, 0) ;
                default: ;
            }
            break ;

        case GB_TIMES_binop_code:

            switch (zcode)
            {
                case GB_INT8_code   : SET (int32_t,  1) ;
                case GB_INT16_code  : SET (int32_t,  1) ;
                case GB_UINT8_code  : SET (uint32_t, 1) ;
                case GB_UINT16_code : SET (uint32_t, 1) ;
                default: ;
            }
            break ;

        case GB_PLUS_binop_code:
        case GB_ANY_binop_code:
        case GB_BOR_binop_code:
        case GB_BXOR_binop_code:

            switch (zcode)
            {
                case GB_INT8_code   : SET (int32_t,  0) ;
                case GB_INT16_code  : SET (int32_t,  0) ;
                case GB_UINT8_code  : SET (uint32_t, 0) ;
                case GB_UINT16_code : SET (uint32_t, 0) ;
                default: ;
            }
            break ;

        case GB_LOR_binop_code   : SET (uint32_t, false)   ;
        case GB_LAND_binop_code  : SET (uint32_t, true)  ;
        case GB_LXOR_binop_code  : SET (uint32_t, false)  ;
        case GB_EQ_binop_code    : SET (uint32_t, true)  ;

        case GB_BAND_binop_code:
        case GB_BXNOR_binop_code:

            switch (zcode)
            {
                case GB_UINT8_code  : SET (uint32_t, 0xFF) ;
                case GB_UINT16_code : SET (uint32_t, 0xFFFF) ;
                default: ;
            }
            break ;

        default : ;
    }
}

