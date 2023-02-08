//------------------------------------------------------------------------------
// GB_macrofy_monoid: build macros for a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_monoid  // construct the macros for a monoid
(
    FILE *fp,           // File to write macros, assumed open already
    // inputs:
    int add_ecode,      // binary op as an enum
    int id_ecode,       // identity value as an enum
    int term_ecode,     // terminal value as an enum (<= 28 is terminal)
    GrB_Monoid monoid,  // monoid to macrofy; null if C is iso for GrB_mxm
    // output:
    const char **u_expression
)
{

    GrB_BinaryOp op = (monoid == NULL) ? NULL : monoid->op ;
    const char *ztype_name = (monoid == NULL) ? "void" : op->ztype->name ;
    int zcode = (monoid == NULL) ? 0 : op->ztype->code ;
    size_t zsize = (monoid == NULL) ? 0 : op->ztype->size ;
    GB_Opcode opcode = (monoid == NULL) ? 0 : op->opcode ;

    //--------------------------------------------------------------------------
    // create macros for the additive operator
    //--------------------------------------------------------------------------

    GB_macrofy_binop (fp, "GB_ADD", false, true, add_ecode, op,
        NULL, u_expression) ;

    //--------------------------------------------------------------------------
    // create macros for the identity value
    //--------------------------------------------------------------------------

    bool has_byte ;
    uint8_t byte ;
    if (monoid == NULL)
    {
        // no values computed (C is iso)
        fprintf (fp, "#define GB_DECLARE_MONOID_IDENTITY(modifier,z)\n") ;
    }
    else if (id_ecode <= 28)
    {
        // built-in monoid: a simple assignment
        const char *id_val = GB_charify_id (id_ecode, zsize, &has_byte, &byte) ;
        fprintf (fp, "#define GB_DECLARE_MONOID_IDENTITY(modifier,z) modifier");
        if (zcode == GB_FC32_code)
        {
            fprintf (fp, " %s z = GxB_CMPLXF (%s,0) ;\n", ztype_name, id_val) ;
        }
        else if (zcode == GB_FC64_code)
        {
            fprintf (fp, " %s z = GxB_CMPLX (%s,0) ;\n", ztype_name, id_val) ;
        }
        else
        {
            fprintf (fp, " %s z = %s ;\n", ztype_name, id_val) ;
        }
        if (has_byte)
        {
            fprintf (fp, "#define GB_HAS_IDENTITY_BYTE 1\n") ;
            fprintf (fp, "#define GB_IDENTITY_BYTE 0x%02x\n", (int) byte) ;
        }
    }
    else
    {
        // user-defined monoid:  all we have are the bytes
        GB_macrofy_bytes (fp, "MONOID_IDENTITY", "zidentity",
            ztype_name, (uint8_t *) (monoid->identity), zsize, true) ;
    }

    //--------------------------------------------------------------------------
    // create macros for the terminal value and terminal conditions
    //--------------------------------------------------------------------------

    bool is_any_monoid = (term_ecode == 18) ;
    if (is_any_monoid)
    {
        // ANY monoid is terminal but with no specific terminal value
        fprintf (fp, "#define GB_IS_ANY_MONOID 1\n") ;
//      set by GB_monoid_shared_definitions.h:
//      fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL" "(modifier,zterminal)\n") ;
//      fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
//      fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) 1\n") ;
//      fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) break\n") ;
    }
    else if (monoid == NULL || monoid->terminal == NULL)
    {
        // monoid is not terminal (either built-in or user-defined)
//      set by GB_monoid_shared_definitions.h:
//      fprintf (fp, "#define GB_IS_ANY_MONOID 0\n") ;
//      fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL(modifier,zterminal)\n") ;
//      fprintf (fp, "#define GB_MONOID_IS_TERMINAL 0\n") ;
//      fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) 0\n") ;
//      fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal)\n") ;
    }
    else if (term_ecode <= 28)
    {
        // built-in terminal monoid: terminal value is a simple assignment
//      set by GB_monoid_shared_definitions.h:
//      fprintf (fp, "#define GB_IS_ANY_MONOID 0\n") ;
        fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        const char *term_value = GB_charify_id (term_ecode, zsize, NULL, NULL) ;
        fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL(modifier,zterminal) "
            "modifier %s zterminal = (%s) (%s) ;\n",
            ztype_name, ztype_name, term_value) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
            "((z) == ((%s) (%s)))\n", ztype_name, term_value) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
            "if ((z) == ((%s) (%s))) break\n", ztype_name, term_value) ;
    }
    else
    {
        // user-defined terminal monoid
//      set by GB_monoid_shared_definitions.h:
//      fprintf (fp, "#define GB_IS_ANY_MONOID 0\n") ;
        fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        GB_macrofy_bytes (fp, "MONOID_TERMINAL", "zterminal",
            ztype_name, monoid->terminal, zsize, false) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
            " (memcmp (&(z), &(zterminal), %d) == 0)\n", (int) zsize) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
            " if (memcmp (&(z), &(zterminal), %d) == 0) break\n",
            (int) zsize) ;
    }

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    if (opcode == GB_EQ_binop_code)
    {
        // EQ monoid
        fprintf (fp, "#define GB_IS_EQ_MONOID 1\n") ;
    }

    if (opcode == GB_PLUS_binop_code && zcode == GB_FC32_code)
    {
        // PLUS_FC32 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC32_MONOID 1\n") ;
    }

    if (opcode == GB_PLUS_binop_code && zcode == GB_FC64_code)
    {
        // PLUS_FC64 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC64_MONOID 1\n") ;
    }

    if (opcode == GB_ANY_binop_code && zcode == GB_FC32_code)
    {
        // ANY_FC32 monoid
        fprintf (fp, "#define GB_IS_ANY_FC32_MONOID 1\n") ;
    }

    if (opcode == GB_ANY_binop_code && zcode == GB_FC64_code)
    {
        // ANY_FC64 monoid
        fprintf (fp, "#define GB_IS_ANY_FC64_MONOID 1\n") ;
    }

    bool is_integer = (zcode >= GB_INT8_code || zcode <= GB_UINT64_code) ;

    if (opcode == GB_MIN_binop_code && is_integer)
    {
        // IMIN monoid (min with any integer type)
        fprintf (fp, "#define GB_IS_IMIN_MONOID 1\n") ;
    }

    if (opcode == GB_MAX_binop_code && is_integer)
    {
        // IMAX monoid (max with any integer typ)
        fprintf (fp, "#define GB_IS_IMAX_MONOID 1\n") ;
    }

    bool is_float = (zcode == GB_FP32_code || zcode == GB_FP64_code) ;

    if (opcode == GB_MIN_binop_code && is_float)
    {
        // FMIN monoid (min with a real floating-point type)
        fprintf (fp, "#define GB_IS_FMIN_MONOID 1\n") ;
    }

    if (opcode == GB_MAX_binop_code && is_float)
    {
        // FMAX monoid (max with a real floating-point type)
        fprintf (fp, "#define GB_IS_FMAX_MONOID 1\n") ;
    }

    // can ignore overflow in ztype when accumulating the result via the monoid
    // zcode == 0: only when C is iso
    bool ztype_ignore_overflow = (zcode == 0 ||
        zcode == GB_INT64_code || zcode == GB_UINT64_code ||
        zcode == GB_FP32_code  || zcode == GB_FP64_code ||
        zcode == GB_FC32_code  || zcode == GB_FC64_code) ;
    if (ztype_ignore_overflow && !is_any_monoid)
    {
        // if the monoid is ANY, this is set to 1 by
        // GB_monoid_shared_definitions.h, so skip it here
        fprintf (fp, "#define GB_ZTYPE_IGNORE_OVERFLOW 1\n") ;
    }

    //--------------------------------------------------------------------------
    // create macros for the atomic CUDA operator, if available
    //--------------------------------------------------------------------------

    // All built-in monoids are handled, except for the double complex cases of
    // ANY and TIMES.  Those need to be done the same way user-defined monoids
    // are computed.

    char *a = NULL ;
    bool user_monoid_atomically = false ;

    switch (add_ecode)
    {

        // user defined monoid: can apply GB_ADD via atomicCAS if the ztype has 
        // 16, 32, or 64 bits
        case  0 :

            user_monoid_atomically =
                (zsize == sizeof (uint16_t) || 
                 zsize == sizeof (uint32_t) ||
                 zsize == sizeof (uint64_t))  ;
            break ;

        // FIRST, ANY, SECOND: atomic write (not double complex)
        case  1 :
        case  2 :

            switch (zcode)
            {
                case GB_BOOL_code    :
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    : 
                case GB_FC32_code    : a = "GB_cuda_atomic_write" ;
                default              : break ;
            }
            break ;

        // MIN (real only)
        case  3 :
        case  4 :
        case  5 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  : 
                case GB_FP32_code    :
                case GB_FP64_code    : a = "GB_cuda_atomic_min" ;
                default              : break ;
            }
            break ;

        // MAX (real only)
        case  6 :
        case  7 :
        case  8 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  : 
                case GB_FP32_code    :
                case GB_FP64_code    : a = "GB_cuda_atomic_max" ;
                default              : break ;
            }
            break ;

        // PLUS:  all types
        case  9 :
        case 10 :
        case 11 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    :
                case GB_FC32_code    :
                case GB_FC64_code    : a = "GB_cuda_atomic_add" ;
                default              : break ;
            }
            break ;

        // TIMES: all real types, and float complex (but not double complex)
        case 12 : 
        case 14 : 

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    : 
                case GB_FC32_code    : a = "GB_cuda_atomic_times" ;
                default              : break ;
            }
            break ;

        // BOR: z = (x | y), bitwise or,
        // logical LOR (via upscale to uint32_t and BOR)
        case 17 :
        case 19 :

            switch (zcode)
            {
                case GB_BOOL_code    :
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_cuda_atomic_bor" ;
                default              : break ;
            }
            break ;

        // BAND: z = (x & y), bitwise and
        // logical LAND (via upscale to uint32_t and BAND)
        case 18 :
        case 20 :

            switch (zcode)
            {
                case GB_BOOL_code    :
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_cuda_atomic_band" ;
                default              : break ;
            }
            break ;

        // BXOR: z = (x ^ y), bitwise xor, and boolean LXOR
        case 16 :
        case 21 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_cuda_atomic_bxor" ;
                default              : break ;
            }
            break ;

        // BXNOR: z = ~(x ^ y), bitwise xnor, and boolean LXNOR
        case 15 :
        case 22 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_cuda_atomic_bxnor" ;
                default              : break ;
            }
            break ;

        // all other monoids
        default: break ;
    }

    if (monoid == NULL || zcode == 0)
    {

        //----------------------------------------------------------------------
        // C is iso: no values computed so no need for any CUDA atomics
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_HAS_CUDA_ATOMIC 0 /* unused; C is iso */\n") ;

    }
    else if (user_monoid_atomically)
    {

        //----------------------------------------------------------------------
        // user-defined monoid with a type of 16, 32, or 64 bits
        //----------------------------------------------------------------------

        char *cuda_type = NULL ;
        if (zsize == sizeof (uint16_t))
        {
            cuda_type = "unsigned short int" ;
        }
        else if (zsize == sizeof (uint32_t))
        {
            cuda_type = "unsigned int" ;
        }
        else // if (zsize == sizeof (uint64_t))
        {
            cuda_type = "unsigned long long int" ;
        }
        fprintf (fp,
            "#define GB_HAS_CUDA_ATOMIC 1\n"
            "#define GB_CUDA_ATOMIC GB_cuda_atomic_user\n"
            "#define GB_CUDA_ATOMIC_TYPE %s\n"
            "#ifdef GB_CUDA_KERNEL\n"
            "static __device__ __inline__ "
            "void GB_cuda_atomic_user (%s *ptr, %s val)\n"
            "{                                                              \n"
            "    %s *p = (%s *) ptr ;                                       \n"
            "    %s assumed ;                                               \n"
            "    %s old = *p ;                                              \n"
            "    do                                                         \n"
            "    {                                                          \n"
            "        // assume the old value                                \n"
            "        assumed = old ;                                        \n"
            "        // compute the new value                               \n"
            "        %s prior_value = GB_PUN (%s, assumed) ;                \n"
            "        %s new_value ;                                         \n"
            "        GB_ADD (new_value, prior_value, val) ;                 \n"
            "        // modify it atomically:                               \n"
            "        old = atomicCAS (p, assumed, GB_PUN (%s, new_value)) ; \n"
            "    }                                                          \n"
            "    while (assumed != old) ;                                   \n"
            "}                                                              \n"
            "#endif\n",
            ztype_name,                 // GB_CUDA_ATOMIC_TYPE
            ztype_name, ztype_name,     // parameters to GB_cuda_atomic_user
            cuda_type, cuda_type,       // typecast the pointer ptr to cuda_type
            cuda_type, cuda_type,       // cuda_type assumed, old
            ztype_name, ztype_name,     // pun for prior value
            ztype_name,                 // type of new value
            cuda_type) ;                // pun back to cuda_type for atomicCAS

    }
    else if (a == NULL)
    {

        //----------------------------------------------------------------------
        // no CUDA atomic available
        //----------------------------------------------------------------------

        // either built-in (GxB_ANY_FC64_MONOID or GxB_TIMES_FC64_MONOID),
        // or user-defined where the type is not 16, 32, or 64 bits in size

        fprintf (fp, "#define GB_HAS_CUDA_ATOMIC 0\n") ;

    }
    else
    {

        // CUDA atomic available for a built-in monoid
        fprintf (fp, "#define GB_HAS_CUDA_ATOMIC 1\n") ;
        fprintf (fp, "#define GB_CUDA_ATOMIC %s\n", a) ;

        // upscale 8-bit and 16-bit types to 32-bits,
        // all others use their native types
        char *t = "" ;
        switch (zcode)
        {
            case GB_INT8_code    : 
            case GB_INT16_code   : 
            case GB_INT32_code   : t = "int32_t"    ; break ;
            case GB_INT64_code   : t = "int64_t"    ; break ;
            case GB_BOOL_code    : 
            case GB_UINT8_code   : 
            case GB_UINT16_code  : 
            case GB_UINT32_code  : t = "uint32_t"   ; break ;
            case GB_UINT64_code  : t = "uint64_t"   ; break ;
            case GB_FP32_code    : t = "float"      ; break ;
            case GB_FP64_code    : t = "double"     ; break ;
            case GB_FC32_code    : t = "GxB_FC32_t" ; break ;
            case GB_FC64_code    : t = "GxB_FC64_t" ; break ;
            default :;
        }
        fprintf (fp, "#define GB_CUDA_ATOMIC_TYPE %s\n", t) ;
    }
}

