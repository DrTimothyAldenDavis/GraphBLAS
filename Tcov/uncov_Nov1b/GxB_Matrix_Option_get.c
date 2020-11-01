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

        case GxB_HYPER_SWITCH  : GB_cov[4554]++ ;  
// covered (4554): 4

            {
                va_start (ap, field) ;
                double *hyper_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (hyper_switch) ;
                (*hyper_switch) = (double) A->hyper_switch ;
            }
            break ;

        case GxB_SPARSITY_CONTROL  : GB_cov[4555]++ ;  
// NOT COVERED (4555):

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                (*sparsity) = A->sparsity ;
            }
            break ;

        case GxB_SPARSITY_STATUS  : GB_cov[4556]++ ;  
// covered (4556): 2400518

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                (*sparsity) = GB_sparsity (A) ;
            }
            break ;

        case GxB_FORMAT  : GB_cov[4557]++ ;  
// covered (4557): 82970

            {
                va_start (ap, field) ;
                GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (format) ;
                (*format) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            }
            break ;

        case GxB_IS_HYPER  : GB_cov[4558]++ ;  // deprecated; use GxB_SPARSITY_STATUS instead
// covered (4558): 24

            {
                va_start (ap, field) ;
                bool *A_is_hyper = va_arg (ap, bool *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (A_is_hyper) ;
                (*A_is_hyper) = (GB_sparsity (A) == GxB_HYPERSPARSE) ;
            }
            break ;

        default  : GB_cov[4559]++ ;  
// covered (4559): 2

            return (GrB_INVALID_VALUE) ;
    }
    return (GrB_SUCCESS) ;
}

