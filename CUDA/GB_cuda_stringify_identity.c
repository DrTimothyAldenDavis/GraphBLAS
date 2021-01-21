//------------------------------------------------------------------------------
// GB_cuda_stringify_identity: return string for identity value
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

#include "GB.h"
#include "GB_cuda_stringify.h"

#define ID(x) ident = (x)

void GB_cuda_stringify_identity        // return string for identity value
(
    // output:
    char *code_string,      // string with the #define macro
    // input:
    GB_Opcode opcode,       // must be a built-in binary operator from a monoid
    GB_Type_code zcode      // type code used in the opcode we want
)
{
    const char *identity_string ;
    int ecode ;
    GB_cuda_enumify_identity (&ecode, opcode, zcode) ;
    GB_cuda_charify_identity_or_terminal (&identity_string, ecode) ;
    GB_cuda_macrofy_identity (code_string, identity_string) ;
}

void GB_cuda_enumify_identity
(
    // output
    int *ecode,             // enumerated identity value, in range 0 to ... (-1 if fail)
    // input
    GB_Opcode opcode,       // must be a built-in binary operator from a monoid
    GB_Type_code zcode      // type code used in the opcode we want
)
{

    int e = -1 ;

    switch (opcode)
    {

        case GB_PLUS_opcode     : e = 0 ; break ; //  ("0") ;

        case GB_TIMES_opcode    : e = 1 ; break ; //  ("1") ;

        case GB_LAND_opcode     : 
        // case GB_LXNOR_opcode : 
        case GB_EQ_opcode       : 
            e = (zcode == GB_BOOL_code) ? 2 : (-1) ; break ; // ("true") ;

        case GB_LOR_opcode      : 
        case GB_LXOR_opcode     : 
            e = (zcode == GB_BOOL_code) ? 3 : (-1) ; break ; // ("false") ;

        case GB_MIN_opcode :

            switch (zcode)
            {
                case GB_BOOL_code   : e =  2 ; break ; // ("true") ;     // boolean AND
                case GB_INT8_code   : e =  4 ; break ; // ("INT8_MAX") ;
                case GB_INT16_code  : e =  5 ; break ; // ("INT16_MAX") ;
                case GB_INT32_code  : e =  6 ; break ; // ("INT32_MAX") ;
                case GB_INT64_code  : e =  7 ; break ; // ("INT64_MAX") ;
                case GB_UINT8_code  : e =  8 ; break ; // ("UINT8_MAX") ;
                case GB_UINT16_code : e =  9 ; break ; // ("UINT16_MAX") ;
                case GB_UINT32_code : e = 10 ; break ; // ("UINT32_MAX") ;
                case GB_UINT64_code : e = 11 ; break ; // ("UINT64_MAX") ;
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 12 ; break ; // ("INFINITY") ;
                default             : e = -1 ; break ;
            }
            break ;

        case GB_MAX_opcode :

            switch (zcode)
            {
                case GB_BOOL_code   : e =  3 ; break ; // ("false") ;    // boolean OR
                case GB_INT8_code   : e = 13 ; break ; // ("INT8_MIN") ;
                case GB_INT16_code  : e = 14 ; break ; // ("INT16_MIN") ;
                case GB_INT32_code  : e = 15 ; break ; // ("INT32_MIN") ;
                case GB_INT64_code  : e = 16 ; break ; // ("INT64_MIN") ;
                case GB_UINT8_code  : e =  0 ; break ; // ("0") ;
                case GB_UINT16_code : e =  0 ; break ; // ("0") ;
                case GB_UINT32_code : e =  0 ; break ; // ("0") ;
                case GB_UINT64_code : e =  0 ; break ; // ("0") ;
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 17 ; break ; // ("-INFINITY") ;
                default             : e = -1 ; break ;
            }
            break ;

        case GB_ANY_opcode   : e = 0 ; break ; // ("0") ;

        default              : e = -1 ; break ; // invalid operator or type

    }

    (*ecode) = e ;
}


    snprintf (code_string, GB_CUDA_STRLEN, "#define GB_IDENTITY (%s)", ident) ;

}

