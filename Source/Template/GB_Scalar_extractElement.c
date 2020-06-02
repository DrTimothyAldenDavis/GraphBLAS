//------------------------------------------------------------------------------
// GB_Scalar_extractElement_template: x = S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = S, typecasting from the
// type of S to the type of x, as needed.

// Returns GrB_SUCCESS if the GxB_Scalar entry is present, and sets x to its
// value.  Returns GrB_NO_VALUE if the GxB_Scalar is not present, and x is
// unmodified.

// This template constructs GrB_*_extractElement_[TYPE], for both the
// GrB_Matrix and GrB_Vector methods, for each of the 13 built-in types, and
// the _UDT method for all user-defined types.

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry from S
(
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    const GxB_Scalar S          // GxB_Scalar to extract a scalar from
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    GB_WHERE (GB_STR (GB_EXTRACT_ELEMENT) " (x, s)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (S) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_MATRIX_WAIT (S) ;

    GB_RETURN_IF_NULL (x) ;

    // GB_XCODE and S must be compatible
    GB_Type_code scode = S->type->code ;
    if (!GB_code_compatible (GB_XCODE, scode))
    { 
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "entry s of type [%s] cannot be typecast\n"
            "to output scalar x of type [%s]",
            S->type->name, GB_code_string (GB_XCODE)))) ;
    }

    if (S->nzmax == 0)
    { 
        // quick return
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // extract the scalar
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Sp = S->p ;
    if (Sp [1] == 0)
    {
        // the scalar has no entry
        return (GrB_NO_VALUE) ;
    }

    #if (GB_XCODE < GB_UDT_code)
    if (GB_XCODE == scode)
    { 
        // copy the value from S into x, no typecasting, for built-in
        // types only.
        GB_XTYPE *GB_RESTRICT Sx = (GB_XTYPE *) S->x ;
        (*x) = Sx [0] ;
    }
    else
    #endif
    { 
        // typecast the value from S into x
        GB_cast_array ((GB_void *) x, GB_XCODE,
            ((GB_void *) S->x), scode, S->type->size, 1, 1) ;
    }
    return (GrB_SUCCESS) ;
}

#undef GB_EXTRACT_ELEMENT
#undef GB_XTYPE
#undef GB_XCODE

