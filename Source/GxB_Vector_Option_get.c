//------------------------------------------------------------------------------
// GxB_Vector_Option_get: get an option in a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_Option_get      // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    GxB_Option_Field field,         // option to query
    ...                             // return value of the vector option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_Option_get (v, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_SPARSITY :

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                if (GB_IS_HYPERSPARSE (v))
                { 
                    (*sparsity) = GxB_HYPERSPARSE ;
                }
                else if (GB_IS_FULL (v))
                { 
                    (*sparsity) = GxB_FULL ;
                }
                else if (GB_IS_BITMAP (v))
                { 
                    (*sparsity) = GxB_BITMAP ;
                }
                else
                { 
                    (*sparsity) = GxB_SPARSE ;
                }
            }
            break ;

        case GxB_FORMAT : 

            {
                // a GrB_Vector is always stored by-column
                va_start (ap, field) ;
                GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (format) ;
                (*format) = GxB_BY_COL ;
            }
            break ;

        case GxB_IS_HYPER : 

            {
                va_start (ap, field) ;
                bool *v_is_hyper = va_arg (ap, bool *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (v_is_hyper) ;
                (*v_is_hyper) = false ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;

    }
    return (GrB_SUCCESS) ;
}

