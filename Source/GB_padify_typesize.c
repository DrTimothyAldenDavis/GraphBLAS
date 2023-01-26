//------------------------------------------------------------------------------
// GB_padify_typesize: pad the size of a scalar type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// round up the size of a type to 2 bytes, or a multiple of 4 bytes

// FIXME: the result may depend on the platform (CPU vs GPU)

#include "GB.h"
#include "GB_stringify.h"

size_t GB_padify_typesize   // pad the size of a type
(
    size_t size
)
{
    if (size == 0)
    {
        // no type at all
        return (0) ;
    }
    else if (size <= sizeof (uint16_t))
    {
        // 1 byte is padded to 2 bytes
        return (sizeof (uint16_t)) ;
    }
    else if (size <= sizeof (uint32_t))
    {
        // 3 bytes are padded to 4 bytes
        return (sizeof (uint32_t)) ;
    }
    else
    {
        // 4 bytes or more: round up to the nearest multiple of 4 bytes
        size_t e = (size % sizeof (uint32_t)) ;
        size_t pad = (e == 0) ? 0 : (sizeof (uint32_t) - e) ;
        return (size + pad) ;
    }
}

