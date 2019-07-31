//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a MATLAB sparse matrix or GraphBLAS struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

mxArray *gb_export              // return the exported MATLAB matrix or struct
(
    GrB_Matrix *C_handle,       // GrB_Matrix to export and free
    bool kind_is_object         // true if output is struct, false if sparse
)
{

    if (kind_is_object)
    {
        return (gb_export_to_mxstruct (C_handle)) ;
    }
    else
    {
        return (gb_export_to_mxarray (C_handle, true)) ;
    }
}

