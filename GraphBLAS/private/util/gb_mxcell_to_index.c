//------------------------------------------------------------------------------
// gb_mxcell_to_index: convert cell array to index list I or colon expression
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Get a list of indices from a MATLAB cell array.

// I is a cell array.  I contains 0, 1, 2, or 3 items:
//
//      0:   { }    This is the MATLAB ':', like C(:,J), refering to all m rows,
//                  if C is m-by-n.
//      1:   { list }  A 1D list of row indices, like C(I,J) in MATLAB.  If the
//                  list is double, then it contains 1-based indices, in the
//                  range 1 to m if C is m-by-n, so that C(1,1) refers to the
//                  entry in the first row and column of C.  If I is int64 or
//                  uint64, then it contains 0-based indices in the range 0 to
//                  m-1, where C(0,0) is the same entry.
//      2:  { start,fini }  start and fini are scalars (either double, int64,
//                  or uint64).  This defines I = start:fini) in MATLAB index
//                  notation.  Typically, start and fini have type double and
//                  refer to 1-based indexing of C.  int64 or uint64 scalars
//                  are treated as 0-based.
//      3:  { start,inc,fini } start, inc, and fini are scalars (double, int64,
//                  or uint64).  This defines I = start:inc:fini in MATLAB
//                  notation.  The start and fini are 1-based if double,
//                  0-based if int64 or uint64.  inc remains the same
//                  regardless of its type.

#include "gb_matlab.h"

GrB_Index *gb_mxcell_to_index   // return index list I
(
    const mxArray *I_cell,      // MATLAB cell array
    const GrB_Index n,          // dimension of matrix being indexed
    bool *I_allocated,          // true if output array I is allocated
    GrB_Index *ni               // length (I)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (I_cell == NULL || !mxIsCell (I_cell), "internal error 6") ;

    //--------------------------------------------------------------------------
    // get the contents of I_cell
    //--------------------------------------------------------------------------

    int len = mxGetNumberOfElements (I_cell) ;
    CHECK_ERROR (len > 3, "index must be a cell array of length 0 to 3") ;

    bool Item_allocated [3] = { false, false, false } ;
    int64_t Item_len [3] = { 0, 0, 0 } ;
    int64_t Item_max [3] = { -1, -1, -1 } ;
    GrB_Index *Item [3] = { NULL, NULL, NULL } ;

    for (int k = 0 ; k < len ; k++)
    {
        // convert I_cell {k} content to an integer list
        Item [k] = gb_mxarray_to_list (mxGetCell (I_cell, k),
            &Item_allocated [k], &Item_len [k], &Item_max [k]) ;
    }

    //--------------------------------------------------------------------------
    // parse the lists in the cell array
    //--------------------------------------------------------------------------

    GrB_Index *I ;

    if (len == 0)
    {

        //----------------------------------------------------------------------
        // I = { }
        //----------------------------------------------------------------------

        (*ni) = n ;
        (*I_allocated) = false ;
        I = (GrB_Index *) GrB_ALL ;

    }
    else if (len == 1)
    {

        //----------------------------------------------------------------------
        // I = { list }
        //----------------------------------------------------------------------

        (*ni) = Item_len [0] ;
        (*I_allocated) = Item_allocated [0] ;
        I = (GrB_Index *) (Item [0]) ;

    }
    else if (len == 2)
    {

        //----------------------------------------------------------------------
        // I = { start, fini }, defining start:fini
        //----------------------------------------------------------------------

        CHECK_ERROR (Item_len [0] != 1 || Item_len [1] != 1,
            "start and fini must be scalars for start:fini") ;

        I = mxCalloc (3, sizeof (GrB_Index)) ;
        (*I_allocated) = true ;

        I [GxB_BEGIN] = Item [0][0] ;
        I [GxB_END  ] = Item [1][0] ;

        if (Item_allocated [0]) gb_mxfree (& (Item [0])) ;
        if (Item_allocated [1]) gb_mxfree (& (Item [1])) ;

        (*ni) = GxB_RANGE ;

    }
    else // if (len == 3)
    {

        //----------------------------------------------------------------------
        // I = { start, inc, fini }, defining start:inc:fini
        //----------------------------------------------------------------------

        CHECK_ERROR (Item_len [0] != 1 || Item_len [1] != 1 ||
            Item_len [2] != 1,
            "start, inc, and fini must be scalars for start:inc:fini") ;

        I = mxCalloc (3, sizeof (GrB_Index)) ;
        (*I_allocated) = true ;

        I [GxB_BEGIN] = Item [0][0] ;
        I [GxB_END  ] = Item [2][0] ;
        int64_t inc = Item [1][0] ;

        if (Item_allocated [1])
        {
            // the 2nd item in the list is inc, and if it was passed in as
            // a double scalar, it has been decremented.  So increment it to
            // get back to the correct value.
            inc++ ;
        }

        if (Item_allocated [0]) gb_mxfree (& (Item [0])) ;
        if (Item_allocated [1]) gb_mxfree (& (Item [1])) ;
        if (Item_allocated [2]) gb_mxfree (& (Item [2])) ;

        if (inc < 0)
        {
            I [GxB_INC] = (GrB_Index) (-inc) ;
            (*ni) = GxB_BACKWARDS ;
        }
        else
        {
            I [GxB_INC] = (GrB_Index) (inc) ;
            (*ni) = GxB_STRIDE ;
        }
    }

    return (I) ;
}

