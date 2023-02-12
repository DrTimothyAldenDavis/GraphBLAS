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
    bool disable_terminal_condition,    // if true, the monoid is assumed
                        // to be non-terminal.  For the (times, firstj, int64)
                        // semiring, times is normally a terminal monoid, but
                        // it's not worth exploiting in GrB_mxm.
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
        fprintf (fp, "#define GB_DECLARE_IDENTITY(z)\n") ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z)\n") ;
    }
    else if (id_ecode <= 28)
    {
        // built-in monoid: a simple assignment
        const char *id_val = GB_charify_id (id_ecode, zsize, &has_byte, &byte) ;
        #define SLEN (256 + GxB_MAX_NAME_LEN)
        char id [SLEN] ;
        if (zcode == GB_FC32_code)
        {
            snprintf (id, SLEN, "%s z = GxB_CMPLXF (%s,0)",
                ztype_name, id_val) ;
        }
        else if (zcode == GB_FC64_code)
        {
            snprintf (id, SLEN, "%s z = GxB_CMPLX (%s,0)",
                ztype_name, id_val) ;
        }
        else
        {
            snprintf (id, SLEN, "%s z = %s", ztype_name, id_val) ;
        }
        fprintf (fp, "#define GB_DECLARE_IDENTITY(z) %s\n", id) ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z) const %s\n", id) ;
        if (has_byte)
        {
            fprintf (fp, "#define GB_HAS_IDENTITY_BYTE 1\n") ;
            fprintf (fp, "#define GB_IDENTITY_BYTE 0x%02x\n", (int) byte) ;
        }
    }
    else
    {
        // user-defined monoid:  all we have are the bytes
        GB_macrofy_bytes (fp, "IDENTITY", "z",
            ztype_name, (uint8_t *) (monoid->identity), zsize, true) ;
        fprintf (fp, "#define GB_DECLARE_IDENTITY_CONST(z) "
            "GB_DECLARE_IDENTITY(z)\n") ;
    }

    //--------------------------------------------------------------------------
    // create macros for the terminal value and terminal conditions
    //--------------------------------------------------------------------------

    // GB_TERMINAL_CONDITION(z,zterminal) should return true if the value of z
    // has reached its terminal value (zterminal), or false otherwise.  If the
    // monoid is not terminal, then the macro should always return false.  The
    // ANY monoid should always return true.

    // GB_IF_TERMINAL_BREAK(z,zterminal) is a macro containing a full
    // statement.  If the monoid is never terminal, it becomes the empty
    // statement.  Otherwise, it checks the terminal condition and does a
    // "break" if true.

    // GB_DECLARE_TERMINAL_CONST(zterminal) declares the zterminal variable as
    // const.  It is empty if the monoid is not terminal.

    bool monoid_is_terminal = false ;

    bool is_any_monoid = (term_ecode == 18) ;
    if (is_any_monoid)
    {
        // ANY monoid is terminal but with no specific terminal value
        fprintf (fp, "#define GB_IS_ANY_MONOID 1\n") ;
        monoid_is_terminal = true ;
    }
    else if (monoid == NULL || monoid->terminal == NULL
        || disable_terminal_condition)
    {
        // monoid is not terminal (either built-in or user-defined), or
        // its terminal condition is ignored (for (times, firstj, int64),
        // for example).
        monoid_is_terminal = false ;
    }
    else if (term_ecode <= 28)
    {
        // built-in terminal monoid: terminal value is a simple assignment
        monoid_is_terminal = true ;
        fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        const char *term_value = GB_charify_id (term_ecode, zsize, NULL, NULL) ;
        fprintf (fp, "#define GB_DECLARE_TERMINAL_CONST(zterminal) "
            "const %s zterminal = ", ztype_name) ;
        if (zcode == GB_FC32_code)
        {
            fprintf (fp, "GxB_CMPLXF (%s,0)\n", term_value) ;
        }
        else if (zcode == GB_FC64_code)
        {
            fprintf (fp, "GxB_CMPLX (%s,0)\n", term_value) ;
        }
        else
        {
            fprintf (fp, "%s\n", term_value) ;
        }
        fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
            "((z) == %s)\n", term_value) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
            "if ((z) == %s) break\n", term_value) ;
    }
    else
    {
        // user-defined terminal monoid
        monoid_is_terminal = true ;
        fprintf (fp, "#define GB_MONOID_IS_TERMINAL 1\n") ;
        GB_macrofy_bytes (fp, "TERMINAL_CONST", "zterminal",
            ztype_name, monoid->terminal, zsize, false) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(z,zterminal) "
            " (memcmp (&(z), &(zterminal), %d) == 0)\n", (int) zsize) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(z,zterminal) "
            " if (memcmp (&(z), &(zterminal), %d) == 0) break\n",
            (int) zsize) ;
    }

    //--------------------------------------------------------------------------
    // determine the OpenMP #pragma omp reduction(op:z) for this monoid
    //--------------------------------------------------------------------------

    // If not #define'd, the default in GB_monoid_shared_definitions.h is no
    // #pragma.  The pragma is empty if the monoid is terminal, since the simd
    // reduction does not work with a 'break' in the loop.

    bool is_complex = (zcode == GB_FC32_code || zcode == GB_FC64_code) ;

    if (!monoid_is_terminal && !is_complex)
    {
        char *op = NULL ;
        if (opcode == GB_PLUS_binop_code)
        {
            // #pragma omp simd reduction(+:z)
            op = "+" ;
        }
        else if (opcode == GB_LXOR_binop_code || opcode == GB_BXOR_binop_code)
        {
            // #pragma omp simd reduction(^:z)
            op = "^" ;
        }
        else if (opcode == GB_TIMES_binop_code)
        {
            // #pragma omp simd reduction(^:z)
            op = "*" ;
        }
        if (op != NULL)
        {
            // The monoid has a "#pragma omp simd reduction(op:z)" statement.
            // There are other OpenMP reductions that could be exploited, but
            // many are for terminal monoids (logical and bitwise AND, OR).
            // The min/max reductions are not exploited because they are
            // terminal monoids for integers.  For floating-point, the NaN
            // handling may differ, so they are not exploited here either.
            fprintf (fp, "#define GB_PRAGMA_SIMD_REDUCTION_MONOID(z) "
                "GB_PRAGMA_SIMD_REDUCTION (%s,z)\n", op) ;
        }
    }

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    bool is_integer = (zcode >= GB_INT8_code || zcode <= GB_UINT64_code) ;
    bool is_fp_real = (zcode == GB_FP32_code || zcode == GB_FP64_code) ;

    if (opcode == GB_PLUS_binop_code && zcode == GB_FC32_code)
    {
        // PLUS_FC32 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC32_MONOID 1\n") ;
    }
    else if (opcode == GB_PLUS_binop_code && zcode == GB_FC64_code)
    {
        // PLUS_FC64 monoid
        fprintf (fp, "#define GB_IS_PLUS_FC64_MONOID 1\n") ;
    }
    else if (opcode == GB_ANY_binop_code && zcode == GB_FC32_code)
    {
        // ANY_FC32 monoid
        fprintf (fp, "#define GB_IS_ANY_FC32_MONOID 1\n") ;
    }
    else if (opcode == GB_ANY_binop_code && zcode == GB_FC64_code)
    {
        // ANY_FC64 monoid
        fprintf (fp, "#define GB_IS_ANY_FC64_MONOID 1\n") ;
    }
    else if (opcode == GB_MIN_binop_code && is_integer)
    {
        // IMIN monoid (min with any integer type)
        fprintf (fp, "#define GB_IS_IMIN_MONOID 1\n") ;
    }
    else if (opcode == GB_MAX_binop_code && is_integer)
    {
        // IMAX monoid (max with any integer typ)
        fprintf (fp, "#define GB_IS_IMAX_MONOID 1\n") ;
    }
    else if (opcode == GB_MIN_binop_code && is_fp_real)
    {
        // FMIN monoid (min with a real floating-point type)
        fprintf (fp, "#define GB_IS_FMIN_MONOID 1\n") ;
    }
    else if (opcode == GB_MAX_binop_code && is_fp_real)
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
    // create macros for atomics on the CPU
    //--------------------------------------------------------------------------

    // atomic write
    bool has_atomic_write = false ;
    char *ztype_atomic = NULL ;
    if (zcode == 0)
    {
        // C is iso (any_pair symbolic semiring)
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 0\n") ;
    }
    else if (zsize == sizeof (uint8_t))
    {
        // int8_t, uint8_t, and 8-bit user-defined types
        ztype_atomic = "uint8_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 8\n") ;
    }
    else if (zsize == sizeof (uint16_t))
    {
        // int16_t, uint16_t, and 16-bit user-defined types
        ztype_atomic = "uint16_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 16\n") ;
    }
    else if (zsize == sizeof (uint32_t))
    {
        // int32_t, uint32_t, float, and 32-bit user-defined types
        ztype_atomic = "uint32_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 32\n") ;
    }
    else if (zsize == sizeof (uint64_t))
    {
        // int64_t, uint64_t, double, float complex, and 64-bit user types
        ztype_atomic = "uint64_t" ;
        has_atomic_write = true ;
        fprintf (fp, "#define GB_Z_ATOMIC_BITS 64\n") ;
    }

    // atomic write for the ztype
    if (has_atomic_write)
    {
        fprintf (fp, "#define GB_Z_HAS_ATOMIC_WRITE 1\n") ;
        if (zcode == GB_FC32_code || zcode == GB_UDT_code)
        {
            // user-defined types of size 1, 2, 4, or 8 bytes can be written
            // atomically, but must use a pun with ztype_atomic.  float complex
            // should also ztype_atomic.
            fprintf (fp, "#define GB_Z_ATOMIC_TYPE %s\n", ztype_atomic) ;
        }
    }

    // OpenMP atomic update support
    bool is_real = (zcode >= GB_BOOL_code && zcode <= GB_FP64_code) ;
    bool has_atomic_update = false ;
    int omp_atomic_version = 2 ;

    switch (opcode)
    {

        case GB_ANY_binop_code   :
            // the ANY monoid is a special case.  It is done with an atomic
            // write, or no update at all.  The atomic write can be done for
            // float complex (64 bits) but not double complex (128 bits).
            // The atomic update is identical: just an atomic write.
            has_atomic_update = has_atomic_write ;
            break ;

        case GB_BOR_binop_code   :
        case GB_BAND_binop_code  :
        case GB_BXOR_binop_code  :
        case GB_LOR_binop_code   :
        case GB_LAND_binop_code  :
        case GB_LXOR_binop_code  :
            // OpenMP 4.0 atomic, not on MS Visual Studio
            has_atomic_update = true ;
            omp_atomic_version = 4 ;
            break ;

        case GB_BXNOR_binop_code :
        case GB_MIN_binop_code   :
        case GB_MAX_binop_code   :
        case GB_EQ_binop_code    : // LXNOR
            // these monoids can be done via atomic compare/exchange
            has_atomic_update = true ;
            break ;

        case GB_PLUS_binop_code  :
            // even complex can be done atomically
            has_atomic_update = true ;
            break ;

        case GB_TIMES_binop_code :
            // real monoids can be done atomically, not complex
            has_atomic_update = is_real ;
            break ;

        default :
            // all other monoids, including user-defined, cannot be done
            // atomically.  Instead, they must be done in a critical section.
            has_atomic_update = false ;
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

    //--------------------------------------------------------------------------
    // include shared definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_monoid_shared_definitions.h\"\n\n") ;
}

