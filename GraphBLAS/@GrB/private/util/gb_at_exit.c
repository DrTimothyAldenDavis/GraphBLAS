//------------------------------------------------------------------------------
// gb_at_exit: terminate GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

void gb_at_exit ( void )
{
    GrB_finalize ( ) ;
}

