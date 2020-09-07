//------------------------------------------------------------------------------
// GxB_Vector_Option_set: set an option in a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_transpose.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_Option_set      // set an option in a vector
(
    GrB_Vector v,                   // descriptor to modify
    GxB_Option_Field field,         // option to change
    ...                             // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    GB_WHERE (v, "GxB_Vector_Option_set (v, field, value)") ;
    GB_BURBLE_START ("GxB_set") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT_VECTOR_OK (v, "v to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the vector option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_SPARSITY :

            {
                va_start (ap, field) ;
                int sparsity = va_arg (ap, int) ;
                va_end (ap) ;
                if (sparsity <= 0 || sparsity > GxB_AUTO_SPARSITY)
                { 
                    // GxB_DEFAULT is zero, so this is changed to
                    // GxB_AUTO_SPARSITY.
                    sparsity = GxB_AUTO_SPARSITY ;
                }
                // a GrB_Vector cannot be hypersparse
                if (sparsity & GxB_HYPERSPARSE)
                {
                    // disable the hypersparsity flag
                    sparsity &= (GxB_FULL + GxB_BITMAP + GxB_SPARSE) ;
                    // enable the sparse flag instead
                    sparsity |= (GxB_SPARSE) ;
                }
                v->sparsity = sparsity ;
            }
            break ;

        default : 

            GB_ERROR (GrB_INVALID_VALUE,
                "invalid option field [%d], can only be GxB_SPARSITY [%d]",
                (int) field, (int) GxB_SPARSITY) ;

    }

    //--------------------------------------------------------------------------
    // conform the vector to its new desired sparsity structure
    //--------------------------------------------------------------------------

    info = GB_conform ((GrB_Matrix) v, Context) ;
    GB_BURBLE_END ;
    if (info == GrB_SUCCESS) ASSERT_VECTOR_OK (v, "v set", GB0) ;
    return (info) ;
}

