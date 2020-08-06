//------------------------------------------------------------------------------
// GxB_Matrix_Option_get: get an option in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Matrix_Option_get      // gets the current option of a matrix
(
    GrB_Matrix A,                   // matrix to query
    GxB_Option_Field field,         // option to query
    ...                             // return value of the matrix option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_Option_get (A, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_HYPER_SWITCH : 

            {
                va_start (ap, field) ;
                double *hyper_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (hyper_switch) ;
                (*hyper_switch) = (double) A->hyper_switch ;
            }
            break ;

        case GxB_SPARSITY :

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                if (A->h != NULL)
                { 
                    (*sparsity) = GxB_HYPERSPARSE ;
                }
                else if (GB_IS_FULL (A))
                { 
                    (*sparsity) = GxB_FULL ;
                }
                else if (GB_IS_BITMAP (A))
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
                va_start (ap, field) ;
                GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (format) ;
                (*format) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            }
            break ;

        case GxB_IS_HYPER : 

            {
                va_start (ap, field) ;
                bool *A_is_hyper = va_arg (ap, bool *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (A_is_hyper) ;
                (*A_is_hyper) = (A->h != NULL) ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;

    }
    return (GrB_SUCCESS) ;
}
