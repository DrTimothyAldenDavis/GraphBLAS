//------------------------------------------------------------------------------
// gb_mxarray_typecode: return the GraphBLAS type of a MATLAB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

int gb_mxarray_typecode        // return the GB_Type_code of a MATLAB matrix
(
    const mxArray *X
)
{

    if (mxIsComplex (X))
    { 

        switch (mxGetClassID (X))
        {
            case mxSINGLE_CLASS   : return ((int) GB_FC32_code) ;
            case mxDOUBLE_CLASS   : return ((int) GB_FC64_code) ;
            default               : return (-1) ;
        }

    }
    else
    { 

        switch (mxGetClassID (X))
        {
            case mxLOGICAL_CLASS  : return ((int) GB_BOOL_code) ;
            case mxINT8_CLASS     : return ((int) GB_INT8_code) ;
            case mxINT16_CLASS    : return ((int) GB_INT16_code) ;
            case mxINT32_CLASS    : return ((int) GB_INT32_code) ;
            case mxINT64_CLASS    : return ((int) GB_INT64_code) ;
            case mxUINT8_CLASS    : return ((int) GB_UINT8_code) ;
            case mxUINT16_CLASS   : return ((int) GB_UINT16_code) ;
            case mxUINT32_CLASS   : return ((int) GB_UINT32_code) ;
            case mxUINT64_CLASS   : return ((int) GB_UINT64_code) ;
            case mxSINGLE_CLASS   : return ((int) GB_FP32_code) ;
            case mxDOUBLE_CLASS   : return ((int) GB_FP64_code) ;
            default               : return (-1) ;
        }
    }

    return (-1) ;
}

