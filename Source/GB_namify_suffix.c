//------------------------------------------------------------------------------
// GB_namify_suffix: create a suffix for a jitifyed kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

//------------------------------------------------------------------------------
// GB_append_suffix
//------------------------------------------------------------------------------

static inline char *GB_append_suffix (char *p, const char *name, int32_t len)
{
    if (name != NULL && len > 0)
    { 
        (*p++) = '_' ;
        (*p++) = '_' ;
        memcpy ((void *) p, (const void *) name, (size_t) len) ;
        p += len ;
    }
    return (p) ;
}

//------------------------------------------------------------------------------
// GB_namify_suffix
//------------------------------------------------------------------------------

void GB_namify_suffix
(
    // output:
    char *suffix,
    // input:
    bool builtin,
    const char *op1_name, int32_t op1_name_len,
    const char *op2_name, int32_t op2_name_len
)
{
    char *p = suffix ;
    if (!builtin)
    {
        p = GB_append_suffix (p, op1_name, op1_name_len) ;
        p = GB_append_suffix (p, op2_name, op2_name_len) ;
    }
    (*p) = '\0' ;
}

