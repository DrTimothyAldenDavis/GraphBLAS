//------------------------------------------------------------------------------
// GB_monoid_shared_definitions.h: common macros for monoids
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_monoid_shared_defintions.h provides default definitions for all monoids,
// if the special cases have not been #define'd prior to #include'ing this
// file.

//------------------------------------------------------------------------------
// special monoids
//------------------------------------------------------------------------------

// 1 if monoid is ANY
#ifndef GB_IS_ANY_MONOID
#define GB_IS_ANY_MONOID 0
#endif

// 1 if monoid is PLUS_FC32
#ifndef GB_IS_PLUS_FC32_MONOID
#define GB_IS_PLUS_FC32_MONOID 0
#endif

// 1 if monoid is PLUS_FC64
#ifndef GB_IS_PLUS_FC64_MONOID
#define GB_IS_PLUS_FC64_MONOID 0
#endif

// 1 if monoid is ANY_FC32
#ifndef GB_IS_ANY_FC32_MONOID
#define GB_IS_ANY_FC32_MONOID 0
#endif

// 1 if monoid is ANY_FC64
#ifndef GB_IS_ANY_FC64_MONOID
#define GB_IS_ANY_FC64_MONOID 0
#endif

// 1 if monoid is MIN for signed or unsigned integers
#ifndef GB_IS_IMIN_MONOID
#define GB_IS_IMIN_MONOID 0
#endif

// 1 if monoid is MAX for signed or unsigned integers
#ifndef GB_IS_IMAX_MONOID
#define GB_IS_IMAX_MONOID 0
#endif

// 1 if monoid is MIN for float or double
#ifndef GB_IS_FMIN_MONOID
#define GB_IS_FMIN_MONOID 0
#endif

// 1 if monoid is MAX for float or double
#ifndef GB_IS_FMAX_MONOID
#define GB_IS_FMAX_MONOID 0
#endif

//------------------------------------------------------------------------------
// monoid identity & terminal value and conditions, and handling ztype overflow
//------------------------------------------------------------------------------

// by default, monoid has no terminal value
#ifndef GB_DECLARE_MONOID_TERMINAL
#define GB_DECLARE_MONOID_TERMINAL(modifier,zterminal)
#endif

// by default, identity value is not a single repeated byte
#ifndef GB_HAS_IDENTITY_BYTE
#define GB_HAS_IDENTITY_BYTE 0
#endif
#ifndef GB_IDENTITY_BYTE
#define GB_IDENTITY_BYTE (none)
#endif

#if GB_IS_ANY_MONOID

    // by default, the ANY monoid is terminal
    #ifndef GB_MONOID_IS_TERMINAL
    #define GB_MONOID_IS_TERMINAL 1
    #endif
    #ifndef GB_TERMINAL_CONDITION
    #define GB_TERMINAL_CONDITION(z,zterminal) 1
    #endif
    #ifndef GB_IF_TERMINAL_BREAK
    #define GB_IF_TERMINAL_BREAK(z,zterminal) break
    #endif

    // ignore overflow since no numerical values computed
    #ifndef GB_ZTYPE_IGNORE_OVERFLOW
    #define GB_ZTYPE_IGNORE_OVERFLOW 1
    #endif

#else

    // monoids are not terminal unless explicitly declared otherwise
    #ifndef GB_MONOID_IS_TERMINAL
    #define GB_MONOID_IS_TERMINAL 0
    #endif
    #ifndef GB_TERMINAL_CONDITION
    #define GB_TERMINAL_CONDITION(z,zterminal) 0
    #endif
    #ifndef GB_IF_TERMINAL_BREAK
    #define GB_IF_TERMINAL_BREAK(z,zterminal)
    #endif

    // default, do not ignore overflow when replacing z+z+...+z with n*z.
    #ifndef GB_ZTYPE_IGNORE_OVERFLOW
    #define GB_ZTYPE_IGNORE_OVERFLOW 0
    #endif

#endif

