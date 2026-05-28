//------------------------------------------------------------------------------
// gb_get_descriptor: convert gb_descriptor to GrB_Descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define FREE_ALL \
    GrB_Descriptor_free (&desc) ;

#define GB_UTIL
#include "gb_interface.h"

GrB_Info gb_get_descriptor
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc            // gb_descriptor, pointer to static struct
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    ASSERT (desc_handle != NULL) ;
    ASSERT (gbdesc != NULL) ;
    (*desc_handle) = NULL ;

    //--------------------------------------------------------------------------
    // create the GrB_Descriptor if any field is non-default
    //--------------------------------------------------------------------------

    if (gbdesc->nondefault)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
        OK (GrB_Descriptor_set_INT32 (desc, gbdesc->out , GrB_OUTP)) ;
        OK (GrB_Descriptor_set_INT32 (desc, gbdesc->in0 , GrB_INP0)) ;
        OK (GrB_Descriptor_set_INT32 (desc, gbdesc->in1 , GrB_INP1)) ;
        OK (GrB_Descriptor_set_INT32 (desc, gbdesc->mask, GrB_MASK)) ;
        OK (GrB_Descriptor_set_INT32 (desc, gbdesc->axb , GxB_AxB_METHOD)) ;
    }

    (*desc_handle) = desc ;
    return (GrB_SUCCESS) ;
}

