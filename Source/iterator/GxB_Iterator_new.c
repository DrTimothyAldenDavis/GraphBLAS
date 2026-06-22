//------------------------------------------------------------------------------
// GxB_Iterator_new: allocate an iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Iterator_new (GxB_Iterator *iterator)
{
    int header_arena = GrB_DEFAULT ;
    return (GxB_Iterator_new_arena (iterator, header_arena)) ;
}

