//------------------------------------------------------------------------------
// GB_Pending.h: data structure and operations for pending tuples
//------------------------------------------------------------------------------

#include "GB.h"

struct GB_Pending_struct    // list of pending tuples for a matrix or task
{
    int64_t n ;         // number of pending tuples to add to matrix
    int64_t nmax ;      // size of i,j,x
    bool sorted ;       // true if pending tuples are in sorted order
    int64_t *i ;        // row indices of pending tuples
    int64_t *j ;        // col indices of pending tuples; NULL if A->vdim <= 1
    void *x ;           // values of pending tuples
    GrB_Type type ;     // the type of s
    size_t size ;       // type->size
    GrB_BinaryOp op ;   // operator to assemble pending tuples
} ;

#define GB_PENDING_INIT 8

bool GB_Pending_alloc       // create a list of pending tuples
(
    GB_Pending *PHandle,    // output
    GrB_Type type,          // type of pending tuples
    GrB_BinaryOp op,        // operator for assembling pending tuples
    bool is_matrix          // true if Pending->j must be allocated
) ;

bool GB_Pending_realloc     // reallocate a list of pending tuples
(
    GB_Pending *PHandle,    // Pending tuple list to reallocate
    int64_t nnew            // # of new tuples to accomodate
) ;

void GB_Pending_free        // free a list of pending tuples
(
    GB_Pending *PHandle
) ;

bool GB_Pending_merge                   // merge pending tuples from each task
(
    GB_Pending *PHandle,                // input/output
    const GrB_Type type,
    const GrB_BinaryOp op,
    const bool is_matrix,
    const GB_task_struct *TaskList,     // list of subassign tasks
    const int ntasks,                   // number of tasks
    const int nthreads                  // number of threads
) ;

//------------------------------------------------------------------------------
// GB_Pending_add:  add an entry A(i,j) to the list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Compare this function with the CSparse function cs_entry, the essence of
// which is copied below.  A CSparse matrix can be held in either compressed
// sparse column format, or as a list of tuples, but never both.  A GraphBLAS
// matrix can have both components.

// The cs_entry function appends a single entry to the end of the tuple list,
// and it doubles the space if no space is available.  It also augments the
// matrix dimension as needed, which GB_Pending_add does not do.

// This function starts with an initial list that is larger than cs_entry
// (which starts with a list of size 1), and it doubles the size as needed.  If
// A has a single column then the column index is not kept.  Finally, this
// function supports any data type whereas CSparse only allows for double.

// Otherwise the two methods are essentially the same.  The reader is
// encouraged the compare/contrast the unique coding styles used in CSparse and
// this implementation of GraphBLAS.  CSparse is concise; the book provides the
// code commentary: Direct Methods for Sparse Linear Systems, Timothy A. Davis,
// SIAM, Philadelphia, Sept. 2006, http://bookstore.siam.org/fa02 .  Csparse is
// at http://faculty.cse.tamu.edu/davis/publications_files/CSparse.zip .

// If the function succeeds, the matrix is added to the queue if it is not
// already there.

// If the function fails to add the pending tuple, the entire matrix is
// cleared of all entries, all pending tuples, and all zombies; and it is
// removed from the queue if it is already there.

// This function is agnostic about the CSR/CSC format of A.  Regardless of the
// format, i refers to an index into the vectors, and j is a vector.  So for
// CSC, i is a row index and j is a column index.  For CSR, i is a column index
// and j is a row index.  This function also does not need to know if A is
// hypersparse or not.

// cs_entry (c)2006-2016, T. A. Davis, included here with the GraphBLAS license

//  /* add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
//  int cs_entry (cs *T, int64_t i, int64_t j, double scalar)
//  {
//      if (!CS_TRIPLET (T) || i < 0 || j < 0) return (0) ;
//      if (T->nz >= T->nzmax && !cs_sprealloc (T,2*(T->nzmax))) return (0) ;
//      if (T->x) T->x [T->nz] = scalar ;
//      T->i [T->nz] = i ;
//      T->p [T->nz++] = j ;
//      T->m = CS_MAX (T->m, i+1) ;
//      T->n = CS_MAX (T->n, j+1) ;
//      return (1) ;
//  }

static inline bool GB_Pending_add   // add a tuple to the list
(
    GB_Pending *PHandle,        // Pending tuples to create or append
    const GB_void *scalar,      // scalar to add to the pending tuples
    const GrB_Type type,        // scalar type, if list is created
    const GrB_BinaryOp op,      // new Pending->op, if list is created
    const int64_t i,            // index into vector
    const int64_t j,            // vector index
    const bool is_matrix        // allocate Pending->j, if list is created
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;
    GB_Pending Pending = (*PHandle) ;
    GrB_Info info = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // allocate the Pending tuples, or ensure existing list is large enough
    //--------------------------------------------------------------------------

    int64_t n = 0 ;

    if (Pending == NULL)
    { 
        // this is the first pending tuple: define the type of the pending
        // tuples, and the operator to eventually be used to assemble them.
        // If op is NULL, the implicit SECOND_Atype operator will be used.
        if (!GB_Pending_alloc (PHandle, type, op, is_matrix)) return (false) ;
        Pending = (*PHandle) ;
    }
    else
    {
        n = Pending->n ;
        if (n == Pending->nmax)
        { 
            // reallocate the list so it can hold the new tuple
            if (!GB_Pending_realloc (PHandle, 1)) return (false) ;
        }
    }

    ASSERT (Pending->type == type) ;
    ASSERT (Pending->nmax > 0 && n < Pending->nmax) ;
    ASSERT (Pending->i != NULL && Pending->x != NULL) ;
    ASSERT ((is_matrix) == (Pending->j != NULL)) ;

    //--------------------------------------------------------------------------
    // keep track of whether or not the pending tuples are already sorted
    //--------------------------------------------------------------------------

    int64_t *restrict Pending_i = Pending->i ;
    int64_t *restrict Pending_j = Pending->j ;

    if (n > 0 && Pending->sorted)
    { 
        int64_t ilast = Pending_i [n-1] ;
        int64_t jlast = (Pending_j != NULL) ? Pending_j [n-1] : 0 ;
        Pending->sorted = (jlast < j) || (jlast == j && ilast <= i) ;
    }

    //--------------------------------------------------------------------------
    // add the (i,j,scalar) or just (i,scalar) if Pending->j is NULL
    //--------------------------------------------------------------------------

    Pending_i [n] = i ;
    if (Pending_j != NULL)
    { 
        Pending_j [n] = j ;
    }
    size_t size = type->size ;
    memcpy (Pending->x +(n*size), scalar, size) ;
    Pending->n++ ;

    return (true) ;     // success
}

