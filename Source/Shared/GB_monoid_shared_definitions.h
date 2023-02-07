//------------------------------------------------------------------------------
// GB_monoid_shared_defintions.h: common macros for monoids
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_monoid_shared_defintions.h provides default definitions for all monoids,
// if the special cases have not been #define'd prior to #include'ing this
// file.

// true if monoid is ANY (update can skipped entirely)
#ifndef GB_IS_ANY_MONOID
#define GB_IS_ANY_MONOID 0
#endif

// by default, monoid has no terminal value
#ifndef GB_DECLARE_MONOID_TERMINAL
#define GB_DECLARE_MONOID_TERMINAL(modifier,zterminal)
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

#endif

