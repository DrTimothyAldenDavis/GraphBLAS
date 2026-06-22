//------------------------------------------------------------------------------
// GrB_Descriptor_new: create a new descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Default values are set to GxB_DEFAULT

#include "GB.h"

GrB_Info GrB_Descriptor_new     // create a new descriptor
(
    GrB_Descriptor *descriptor  // handle of descriptor to create
)
{
    int header_arena = GrB_DEFAULT ;
    return (GxB_Descriptor_new_arena (descriptor, header_arena)) ;
}

