//------------------------------------------------------------------------------
// gb_mxstring_to_format: get the format from a MATLAB string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Valid format strings:

//  'by row'            sparsity is GxB_AUTO_SPARSITY for these 2 strings
//  'by col'

//  'sparse by row'
//  'hypersparse by row'
//  'bitmap by row'
//  'full by row'

//  'sparse by col'
//  'hypersparse by col'
//  'bitmap by col'
//  'full by col'

//  'sparse'            fmt is GxB_BY_COL for these four strings
//  'hypersparse'
//  'bitmap'
//  'full'

#include "gb_matlab.h"

bool gb_mxstring_to_format      // true if a valid format is found
(
    // input
    const mxArray *mxformat,    // MATLAB string, 'by row' or 'by col'
    // output
    GxB_Format_Value *fmt,
    int *sparsity
)
{

    (*fmt) = GxB_BY_COL ;
    (*sparsity) = GxB_AUTO_SPARSITY ;
    #define LEN 256
    char format_string [LEN+2] ;
    gb_mxstring_to_string (format_string, LEN, mxformat, "format") ;

    if (MATCH (format_string, "by row"))
    { 
        (*fmt) = GxB_BY_ROW ;
    }
    else if (MATCH (format_string, "by col"))
    { 
        ;
    }
    else if (MATCH (format_string, "sparse") ||
             MATCH (format_string, "sparse by col"))
    { 
        (*sparsity) = GxB_SPARSE ;
    }
    else if (MATCH (format_string, "hypersparse") ||
             MATCH (format_string, "hypersparse by col"))
    { 
        (*sparsity) = GxB_HYPERSPARSE ;
    }
    else if (MATCH (format_string, "bitmap") ||
             MATCH (format_string, "bitmap by col"))
    { 
        (*sparsity) = GxB_BITMAP ;
    }
    else if (MATCH (format_string, "full") ||
             MATCH (format_string, "full by col"))
    { 
        (*sparsity) = GxB_FULL + GxB_BITMAP ;
    }
    else if (MATCH (format_string, "sparse by row"))
    { 
        (*sparsity) = GxB_SPARSE ;
        (*fmt) = GxB_BY_ROW ;
    }
    else if (MATCH (format_string, "hypersparse by row"))
    { 
        (*sparsity) = GxB_HYPERSPARSE ;
        (*fmt) = GxB_BY_ROW ;
    }
    else if (MATCH (format_string, "bitmap by row"))
    { 
        (*sparsity) = GxB_BITMAP ;
        (*fmt) = GxB_BY_ROW ;
    }
    else if (MATCH (format_string, "full by row"))
    { 
        (*sparsity) = GxB_FULL + GxB_BITMAP ;
        (*fmt) = GxB_BY_ROW ;
    }
    else
    { 
        // The string is not a format string, but this is not an error here.
        // For example, G = GrB (m,n,'double','by row') queries both its string
        // input arguments with this function and gb_mxstring_to_type, to parse
        // its inputs.
        return (false) ;
    }

    return (true) ;
}

