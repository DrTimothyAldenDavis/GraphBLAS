//------------------------------------------------------------------------------
// GB_Iterator_rc_attach: attach a row/col iterator to matrix and seek to A(:,j)
//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GB_PUBLIC
GrB_Info GB_Iterator_rc_attach
(
    GxB_Iterator iterator,
    // input
    GrB_Matrix A,
    GrB_Index j,
    bool kth_vector,
    GxB_Format_Value format,
    GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (iterator) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    if ((format == GxB_BY_ROW &&  A->is_csc) ||
        (format == GxB_BY_COL && !A->is_csc))
    {
        return (GrB_NOT_IMPLEMENTED) ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work on the matrix
    //--------------------------------------------------------------------------

    if (GB_ANY_PENDING_WORK (A))
    {
        GrB_Info info ;
        GB_CONTEXT ("GxB_Iterator_attach") ;
        if (desc != NULL)
        {
            // get the # of threads to use
            Context->nthreads_max = desc->nthreads_max ;
            Context->chunk = desc->chunk ;
        }
        GB_OK (GB_wait (A, "A", Context)) ;
    }

    //--------------------------------------------------------------------------
    // get the matrix and save its contents in the iterator
    //--------------------------------------------------------------------------

    iterator->pmax = GB_nnz_held (A) ;
    iterator->avlen = A->vlen ;
    iterator->avdim = A->vdim ;
    iterator->anvec = A->nvec ;
    iterator->Ap = A->p ;
    iterator->Ah = A->h ;
    iterator->Ab = A->b ;
    iterator->Ai = A->i ;
    iterator->Ax = A->x ;
    iterator->type_size = A->type->size ;
    iterator->A_sparsity = GB_sparsity (A) ;
    iterator->iso = A->iso ;
    iterator->by_col = A->is_csc ;

    //--------------------------------------------------------------------------
    // attach the iterator to A(:,j)
    //--------------------------------------------------------------------------

    return (GB_Iterator_rc_seek (iterator, j, kth_vector)) ;
}

