//------------------------------------------------------------------------------
// gb_mxarray_to_descriptor: get the contents of a GraphBLAS Descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// get a GraphBLAS descriptor from a MATLAB struct.

#include "gb_matlab.h"

#define LEN 100

static void get_descriptor
(
    GrB_Descriptor D,               // GraphBLAS descriptor to modify
    const mxArray *D_matlab,        // MATLAB struct with d.out, etc
    const char *fieldname,          // fieldname to extract from D_matlab
    const GrB_Desc_Field field      // field to set in D
)
{

    // find the field in the MATLAB struct
    int fieldnumber = mxGetFieldNumber (D_matlab, fieldname) ;
    if (fieldnumber >= 0)
    {

        // the field is present
        mxArray *value = mxGetFieldByNumber (D_matlab, 0, fieldnumber) ;

        if (MATCH (fieldname, "nthreads"))
        {

            // nthreads must be a numeric scalar
            CHECK_ERROR (!gb_mxarray_is_scalar (value),
                "d.nthreads must be a scalar") ;
            int nthreads_max = (int) mxGetScalar (value) ;
            OK (GxB_set (D, GxB_NTHREADS, nthreads_max)) ;

        }
        else if (MATCH (fieldname, "chunk"))
        {

            // chunk must be a numeric scalar
            CHECK_ERROR (!gb_mxarray_is_scalar (value),
                "d.chunk must be a scalar") ;
            double chunk = mxGetScalar (value) ;
            OK (GxB_set (D, GxB_CHUNK, chunk)) ;

        }
        else
        {

            // get the string from the MATLAB field
            char s [LEN+2] ;
            gb_mxstring_to_string (s, LEN, value, "field") ;

            // convert the string to a Descriptor value, and set the value
            if (MATCH (s, "default"))
            {
                OK (GxB_set (D, field, GxB_DEFAULT)) ;
            }
            else if (MATCH (s, "transpose"))
            {
                OK (GxB_set (D, field, GrB_TRAN)) ;
            }
            else if (MATCH (s, "complement"))
            {
                OK (GxB_set (D, field, GrB_SCMP)) ;
            }
            else if (MATCH (s, "replace"))
            {
                OK (GxB_set (D, field, GrB_REPLACE)) ;
            }
            else if (MATCH (s, "gustavson"))
            {
                OK (GxB_set (D, field, GxB_AxB_GUSTAVSON)) ;
            }
            else if (MATCH (s, "dot"))
            {
                OK (GxB_set (D, field, GxB_AxB_DOT)) ;
            }
            else if (MATCH (s, "heap"))
            {
                OK (GxB_set (D, field, GxB_AxB_HEAP)) ;
            }
            else
            {
                // the string must be one of the strings listed above
                ERROR ("unrecognized descriptor value") ;
            }
        }
    }
}

//------------------------------------------------------------------------------
// gb_mxarray_to_descriptor
//------------------------------------------------------------------------------

GrB_Descriptor gb_mxarray_to_descriptor     // return a new descriptor
(
    const mxArray *D_matlab,        // MATLAB struct
    bool *kind_is_object            // descriptor.kind = 'object' or 'sparse'
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // a null descriptor is OK; the method will use defaults
    if (gb_mxarray_is_empty (D_matlab))
    {
        return (NULL) ;
    }

    // if present, the MATLAB D must be a struct
    CHECK_ERROR (!mxIsStruct (D_matlab), "descriptor must be a struct") ;

    //--------------------------------------------------------------------------
    // create the GraphBLAS descriptor
    //--------------------------------------------------------------------------

    GrB_Descriptor D ;
    OK (GrB_Descriptor_new (&D)) ;

    // get each component of the descriptor struct
    get_descriptor (D, D_matlab, "out"     , GrB_OUTP) ;
    get_descriptor (D, D_matlab, "in0"     , GrB_INP0) ;
    get_descriptor (D, D_matlab, "in1"     , GrB_INP1) ;
    get_descriptor (D, D_matlab, "mask"    , GrB_MASK) ;
    get_descriptor (D, D_matlab, "axb"     , GxB_AxB_METHOD) ;
    get_descriptor (D, D_matlab, "nthreads", GxB_NTHREADS) ;
    get_descriptor (D, D_matlab, "chunk"   , GxB_CHUNK) ;

    //--------------------------------------------------------------------------
    // get the desired kind of output
    //--------------------------------------------------------------------------

    // By default, gbmxm (...) returns a MATLAB sparse matrix.
    // gb.mxm (...) returns an object.

    (*kind_is_object) = false ;

    mxArray *mxkind = mxGetField (D_matlab, 0, "kind") ;
    if (mxkind != NULL)
    {
        // get the string from the MATLAB field
        char s [LEN+2] ;
        gb_mxstring_to_string (s, LEN, mxkind, "kind") ;
        if (MATCH (s, "object"))
        {
            (*kind_is_object) = true ;
        }
        else if (MATCH (s, "sparse"))
        {
            (*kind_is_object) = false ;
        }
        else
        {
            ERROR ("invalid descriptor.kind") ;
        }
    }

    //--------------------------------------------------------------------------
    // return the non-null Descriptor to the caller
    //--------------------------------------------------------------------------

    return (D) ;
}

