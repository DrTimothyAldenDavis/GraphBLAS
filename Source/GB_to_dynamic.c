//------------------------------------------------------------------------------
// GB_to_dynamic: ensure a matrix has a dynamic header, not static
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The A_input matrix has either a dynamic or static header.  If it has a
// dynamic header, a purely shallow copy of A_input is created and returned as
// &A.  If A already has a dynamic header, then the output matrix &A is just
// A_input.  The A_input matrix is not modified in either case.

// If A_input has a static header but has pending work, it is not safe to use
// GrB_Matrix_wait on the result &A, because A is shallow.  The caller must do
// the wait on A_input first, or it must use A_input and A as-is, with no
// changes.  For example, if the algorithm is tolerant of a jumbled matrix, it
// can use A_input or A as-is, even though A is a shallow copy of A_input.

// Once a method is done with A, it should free it, as follows.  This has no
// effect on A_input, since all content of A is shallow:

//      GrB_Matrix A = NULL ;
//      bool A_input_is_static = false ;
//      GB_to_dynamic (&A, &A_input_is_static, A_input, Context) ;
//      ... use A instead of A_input ...
//      if (A_input_is_static) GB_Matrix_free (&A) ;

#include "GB_dynamic.h"

GrB_Info GB_to_dynamic
(
    // output
    GrB_Matrix *Ahandle,        // output matrix A with dynamic header
    bool *A_input_is_static,    // if true, A_input has a static header
    // input
    GrB_Matrix A_input,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;
    ASSERT (A_input_is_static != NULL) ;
    ASSERT (Ahandle != NULL) ;
    (*A_input_is_static) = false ;
    (*Ahandle) = NULL ;

    //--------------------------------------------------------------------------
    // quick return if no input matrix
    //--------------------------------------------------------------------------

    if (A_input == NULL)
    {
        // nothing to do; this is not an error condition, since A_input might
        // be an optional mask matrix
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // convert A_input to a dynamic header, if necessary 
    //--------------------------------------------------------------------------

    (*A_input_is_static) = A_input->static_header ;
    if (*A_input_is_static)
    {
        // allocate a new dynamic header for A and copy A_input into it
        size_t header_size ;
        A = GB_MALLOC (1, struct GB_Matrix_opaque, &header_size) ;
        if (A == NULL)
        {
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        // copy the contents of the static header into the dynamic header
        memcpy (A, A_input, sizeof (struct GB_Matrix_opaque)) ;
        A->static_header = false ;
        A->header_size = header_size ;
        A->p_shallow = (A->p != NULL)  ;
        A->h_shallow = (A->h != NULL)  ;
        A->b_shallow = (A->b != NULL)  ;
        A->i_shallow = (A->i != NULL)  ;
        A->x_shallow = (A->x != NULL)  ;
    }
    else
    {
        // A_input already has a dynamic header, so A is just A_input
        A = A_input ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*Ahandle) = A ;
    return (GrB_SUCCESS) ;
}

