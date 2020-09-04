//------------------------------------------------------------------------------
// GB_bitmap_selector:  select entries from a bitmap or full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_select.h"
#include "GB_sel__include.h"

#define GB_FREE_ALL ;

GrB_Info GB_bitmap_selector
(
    GrB_Matrix *Chandle,        // output matrix, NULL to modify A in-place
    GB_Select_Opcode opcode,    // selector opcode
    const GxB_select_function user_select,      // user select function
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const int64_t ithunk,       // (int64_t) Thunk, if Thunk is NULL
    const GB_void *GB_RESTRICT xthunk,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (GB_is_packed (A)) ;
    ASSERT (opcode != GB_RESIZE_opcode) ;
    ASSERT (opcode != GB_NONZOMBIE_opcode) ;
    ASSERT_MATRIX_OK (A, "A for bitmap selector", GB0) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t anz = GB_NNZ_HELD (A) ;
    const GB_Type_code typecode = A->type->code ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    // TODO: can malloc C->b
    // TODO: must calloc C->x for GB_EQ_ZERO_opcode

    GrB_Matrix C = NULL ;
    GB_OK (GB_new_bix (&C, // always bitmap
        A->type, A->vlen, A->vdim, GB_Ap_calloc, true,
        GxB_BITMAP, A->hyper_switch, -1, anz, true, Context)) ;
    int64_t cnvals ;

    // if (opcode == GB_EQ_ZERO_opcode)
    { 
        // TODO use calloc instead
        memset (C->x, 0, anz * A->type->size) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // launch the switch factory to select the entries
    //--------------------------------------------------------------------------

    #define GB_BITMAP_SELECTOR
    #define GB_selbit(opname,aname) GB_sel_bitmap_ ## opname ## aname
    #define GB_SEL_WORKER(opname,aname,atype)                           \
    {                                                                   \
        GB_selbit (opname, aname) (C->b, C->x, &cnvals, A, flipij,      \
            ithunk, (atype *) xthunk, user_select, nthreads) ;          \
    }                                                                   \
    break ;
    #include "GB_select_factory.c"

    //--------------------------------------------------------------------------
    // create the result
    //--------------------------------------------------------------------------

    if (Chandle == NULL)
    {

        //----------------------------------------------------------------------
        // transplant C back into A
        //----------------------------------------------------------------------

        GB_phbix_free (A) ;
        if (C->nvals == 0)
        { 
            GB_FREE (C->b) ;
            GB_FREE (C->x) ;
            A->nzmax = 0 ;
        }
        A->b = C->b ;
        C->b = NULL ;
        A->x = C->x ;
        C->x = NULL ;
        A->nvals = cnvals ;
        A->nzmax = C->nzmax ;
        A->magic = GB_MAGIC ;
        GB_Matrix_free (&C) ;
        ASSERT_MATRIX_OK (A, "A in place from bitmap selector", GB0) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // return C as (*Chandle)
        //----------------------------------------------------------------------

        (*Chandle) = C ;
        C->nvals = cnvals ;
        ASSERT_MATRIX_OK (C, "C from bitmap selector", GB0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

