//------------------------------------------------------------------------------
// gb_mxarray_to_indices: convert a list of indices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Get a list of indices from a MATLAB array.

#include "gbmex.h"

void gb_mxarray_to_indices      // convert a list of indices
(
    GrB_Index **I_result,       // index array returned
    const mxArray *I_matlab,    // MATLAB mxArray to get
    GrB_Index *ni,              // length of I, or special
    GrB_Index Icolon [3],       // for all but GB_LIST
    bool *I_is_list,            // true if GB_LIST
    bool *I_is_allocated,       // true if index array was allocated
    int64_t *I_max              // largest entry 
)
{

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    mxArray *X ;
    GrB_Index *I ;

    (*I_result) = NULL ;
    (*I_is_allocated) = false ;

    Icolon [0] = 0 ;
    Icolon [1] = 0 ;
    Icolon [2] = 0 ;

    //--------------------------------------------------------------------------
    // I can be NULL, a struct, uint64 array, or double array
    //--------------------------------------------------------------------------

    if (I_matlab == NULL)
    {

        //----------------------------------------------------------------------
        // I is NULL: GrB_ALL
        //----------------------------------------------------------------------

        (*ni) = 0 ;                             // unused
        (*I_result) = (GrB_Index *) GrB_ALL ;   // like the ":" in C=A(:,j)
        (*I_is_list) = false ;

    }
    else if (mxIsStruct (I_matlab))
    {

        //----------------------------------------------------------------------
        // I is a struct with 3 integers: begin, inc, end
        //----------------------------------------------------------------------

        (*I_is_list) = false ;

        // look for I.begin (required)
        X = mxGetField (I_matlab, 0, "begin") ;
        CHECK_ERROR (X == NULL, ".begin missing") ;
        CHECK_ERROR (!IS_SCALAR (X), ".begin must be a scalar") ;
        Icolon [GxB_BEGIN] = (GrB_Index) mxGetScalar (X) ;

        // look for I.end (required)
        X = mxGetField (I_matlab, 0, "end") ;
        CHECK_ERROR (X == NULL, ".end missing") ;
        CHECK_ERROR (!IS_SCALAR (X), ".end must be a scalar") ;
        Icolon [GxB_END] = (GrB_Index) mxGetScalar (X) ;

        // look for I.inc (optional)
        X = mxGetField (I_matlab, 0, "inc") ;
        if (X == NULL)
        {
            // colon notation: I represents begin:end
            (*ni) = GxB_RANGE ;
            Icolon [GxB_INC] = 1 ;
        }
        else
        {
            // I.inc is present: I represents begin:inc:end
            CHECK_ERROR (!IS_SCALAR (X), ".end must be a scalar") ;
            int64_t inc = (int64_t) mxGetScalar (X) ;
            if (inc >= 0)
            {
                (*ni) = GxB_STRIDE ;
                Icolon [GxB_INC] = (GrB_Index) inc ;
            }
            else
            {
                // GraphBLAS must be given the magnitude of the stride
                (*ni) = GxB_BACKWARDS ;
                Icolon [GxB_INC] = (GrB_Index) (-inc) ;
            }
        }
        (*I_result) = Icolon ;

    }
    else if (mxIsClass (I_matlab, "uint64"))
    {

        //----------------------------------------------------------------------
        // I is a uint64 array; returned as a shallow pointer and used as-is
        //----------------------------------------------------------------------

        (*I_is_list) = true ;
        (*ni) = (uint64_t) mxGetNumberOfElements (I_matlab) ;
        (*I_result) = mxGetUint64s (I_matlab) ;

    }
    else if (mxIsClass (I_matlab, "double") && !mxIsComplex (I_matlab))
    {

        //----------------------------------------------------------------------
        // I is a double array: typecast and convert from 1-based to 0-based
        //----------------------------------------------------------------------

        (*I_is_list) = true ;
        double *I_double = mxGetDoubles (I_matlab) ;
        int64_t len = mxGetNumberOfElements (I_matlab) ;
        (*ni) = (uint64_t) len ;

        GrB_Index *I = (GrB_Index *) mxMalloc ((*ni+1) * sizeof (GrB_Index)) ;
        (*I_result) = I ;
        (*I_is_allocated) = true ;

        // convert from 1-based to 0-based indices
        // TODO do this in parallel
        for (int64_t k = 0 ; k < len ; k++)
        {
            I [k] = ((uint64_t) I_double [k]) - 1 ;
            // printf ("index [%g] = %g %g\n", (double) k, (double) I [k],
                // I_double [k]) ;
        }

    }
    else
    {
        ERROR ("invalid index array") ;
    }
}

