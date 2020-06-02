//------------------------------------------------------------------------------
// GB_Vector_extractElement: x = V(row)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = V(row), typecasting from the
// type of V to the type of x, as needed.

// Returns GrB_SUCCESS if V(row) is present, and sets x to its value.
// Returns GrB_NO_VALUE if V(row) is not present, and x is unmodified.

// This template constructs GrB_Vector_extractElement_[TYPE], for each of the
// 13 built-in types, and the _UDT method for all user-defined types.

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry, x = V(row)
(
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    const GrB_Vector V,         // vector to extract a scalar from
    GrB_Index row               // row index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    GB_WHERE (GB_STR (GB_EXTRACT_ELEMENT) " (x, v, i)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (V) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_VECTOR_WAIT (V) ;

    GB_RETURN_IF_NULL (x) ;

    // look for index i in the GrB_Vector
    int64_t i = row ;

    // check index
    if (row >= V->vlen)
    { 
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG,
            "Index " GBu " out of range; must be < " GBd, row, V->vlen))) ;
    }

    // GB_XCODE and V must be compatible
    GB_Type_code vcode = V->type->code ;
    if (!GB_code_compatible (GB_XCODE, vcode))
    { 
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "entry v(i) of type [%s] cannot be typecast\n"
            "to output scalar x of type [%s]",
            V->type->name, GB_code_string (GB_XCODE)))) ;
    }

    if (V->nzmax == 0)
    { 
        // quick return
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // get the pattern of the vector
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Vp = V->p ;
    const int64_t *GB_RESTRICT Vi = V->i ;
    bool found ;

    // extract from a GrB_Vector
    int64_t pleft = 0 ;
    int64_t pright = Vp [1] - 1 ;

    //--------------------------------------------------------------------------
    // binary search in kth vector for index i
    //--------------------------------------------------------------------------

    // Time taken for this step is at most O(log(nnz(V))).
    GB_BINARY_SEARCH (i, Vi, pleft, pright, found) ;

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found)
    {
        #if (GB_XCODE < GB_UDT_code)
        if (GB_XCODE == vcode)
        { 
            // copy the value from V into x, no typecasting, for built-in
            // types only.
            GB_XTYPE *GB_RESTRICT Vx = (GB_XTYPE *) V->x ;
            (*x) = Vx [pleft] ;
        }
        else
        #endif
        { 
            // typecast the value from V into x
            size_t vsize = V->type->size ;
            GB_cast_array ((GB_void *) x, GB_XCODE,
                ((GB_void *) V->x) +(pleft*vsize), vcode, vsize, 1, 1) ;
        }
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // Entry not found.
        return (GrB_NO_VALUE) ;
    }
}

#undef GB_EXTRACT_ELEMENT
#undef GB_XTYPE
#undef GB_XCODE

