//------------------------------------------------------------------------------
// GB_namify_problem: construct a unique name for a problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_namify_problem
(
    // output:
    char *problem_name,     // of size at least 256 + 8*GxB_MAX_NAME_LEN
    // input:
    char *base_name,
    const int scode_width,  // # of hexadecimal digits to print for scode
    const uint64_t scode,
    const bool builtin,     // true if all objects are builtin
    const char *opname1,    // each string has size at most GxB_MAX_NAME_LEN
    const char *opname2,
    const char *typename1,
    const char *typename2,
    const char *typename3,
    const char *typename4,
    const char *typename5,
    const char *typename6
)
{

    if (builtin)
    {
        // keep the names of built-in operators short
        sprintf (problem_name, "%s%0*" PRIx64, base_name, scode_width, scode) ;
    }
    else
    {
        // at least one type or operator is user-defined
        sprintf (problem_name,
        "%s_%0*" PRIx64 "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
        base_name, scode_width, scode,
        (opname1   == NULL) ? "" : "_", (opname1   == NULL) ? "" : opname1,
        (opname2   == NULL) ? "" : "_", (opname2   == NULL) ? "" : opname2,
        (typename1 == NULL) ? "" : "_", (typename1 == NULL) ? "" : typename1,
        (typename2 == NULL) ? "" : "_", (typename2 == NULL) ? "" : typename2,
        (typename3 == NULL) ? "" : "_", (typename3 == NULL) ? "" : typename3,
        (typename4 == NULL) ? "" : "_", (typename4 == NULL) ? "" : typename4,
        (typename5 == NULL) ? "" : "_", (typename5 == NULL) ? "" : typename5,
        (typename6 == NULL) ? "" : "_", (typename6 == NULL) ? "" : typename6) ;
    }
}

