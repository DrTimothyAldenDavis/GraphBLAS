//------------------------------------------------------------------------------
// gb_default_format: determine the default format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GxB_Format_Value gb_default_format      // GxB_BY_ROW or GxB_BY_COL
(
    GrB_Index nrows,        // row vectors are stored by row
    GrB_Index ncols         // column vectors are stored by column
)
{

    GxB_Format_Value format ;
    if (ncols == 1)
    {
        // column vectors are stored by column, by default
        format = GxB_BY_COL ;
    }
    else if (nrows == 1)
    {
        // row vectors are stored by column, by default
        format = GxB_BY_ROW ;
    }
    else
    {
        // get the default format
        OK (GxB_get (GxB_FORMAT, &format)) ;
    }
    return (format) ;
}

