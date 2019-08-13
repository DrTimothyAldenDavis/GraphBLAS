//------------------------------------------------------------------------------
// gb_mxarray_to_list: convert a MATLAB array to a list of integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The MATLAB list may be double, int64, or uint64.  If double, a new integer
// list is created, and the 1-based input list is converted to the 0-based
// integer list.

#include "gb_matlab.h"

int64_t *gb_mxarray_to_list     // return List of integers
(
    const mxArray *mxList,      // list to extract
    bool *allocated,            // true if output list was allocated
    int64_t *len,               // length of list
    int64_t *List_max           // max entry in the list, if computed
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (!mxIsNumeric (mxList), "index list must be numeric") ;
    CHECK_ERROR (mxIsSparse (mxList), "index list cannot be sparse") ;

    //--------------------------------------------------------------------------
    // get the length and class of the MATLAB list
    //--------------------------------------------------------------------------

    (*len) = mxGetNumberOfElements (mxList) ;
    mxClassID class = mxGetClassID (mxList) ;

    //--------------------------------------------------------------------------
    // extract the contents and convert to int64_t
    //--------------------------------------------------------------------------

    if (*len == 0)
    {
        (*allocated) = true ;
        return ((int64_t *) mxCalloc (1, sizeof (int64_t))) ;
    }
    else if (class == mxINT64_CLASS)
    {
        // input list is int64; just return a shallow pointer
        (*allocated) = false ;
        return ((int64_t *) mxGetInt64s (mxList)) ;
    }
    else if (class == mxUINT64_CLASS)
    {
        // input list is uint64; just return a shallow pointer
        (*allocated) = false ;
        return ((int64_t *) mxGetUint64s (mxList)) ;
    }
    else if (class == mxDOUBLE_CLASS)
    {
        // allocate an index array and copy double to GrB_Index; also
        // convert from 1-based to 0-based
        int64_t *List = mxMalloc ((*len) * sizeof (int64_t)) ;
        double *List_double = mxGetDoubles (mxList) ;
        (*List_max) = -1 ;
        bool ok = GB_matlab_helper3 (List, List_double, (*len), List_max) ;
        CHECK_ERROR (!ok, "index must be integer") ;
        (*allocated) = true ;
        return (List) ;
    }
    else
    {
        ERROR ("integer array must be double, int64, or uint64") ;
    }
}

