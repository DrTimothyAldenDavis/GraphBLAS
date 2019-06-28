//------------------------------------------------------------------------------
// GB_Pending_alloc: allocate a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_Pending.h"

bool GB_Pending_alloc       // create a list of pending tuples
(
    GB_Pending *PHandle,    // output
    GrB_Type type,          // type of pending tuples
    GrB_BinaryOp op,        // operator for assembling pending tuples
    bool is_matrix          // true if Pending->j must be allocated
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;
    (*PHandle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate the Pending header
    //--------------------------------------------------------------------------

    GB_Pending Pending ;
    GB_CALLOC_MEMORY (Pending, 1, sizeof (struct GB_Pending_struct)) ;
    if (Pending == NULL)
    { 
        // out of memory
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // initialize the contents of the Pending tuples
    //--------------------------------------------------------------------------

    Pending->n = 0 ;                    // no pending tuples yet
    Pending->nmax = GB_PENDING_INIT ;   // initial size of list
    Pending->sorted = true ;            // keep track if tuples are sorted
    Pending->type = type ;              // type of pending tuples
    Pending->size = type->size ;        // size of pending tuple type
    Pending->op = op ;                  // pending operator (NULL is OK)

    GB_MALLOC_MEMORY (Pending->i, GB_PENDING_INIT, sizeof (int64_t)) ;

    if (is_matrix)
    { 
        GB_MALLOC_MEMORY (Pending->j, GB_PENDING_INIT, sizeof (int64_t)) ;
    }
    else
    { 
        Pending->j = NULL ;
    }

    GB_MALLOC_MEMORY (Pending->x, GB_PENDING_INIT, Pending->size) ;

    if (Pending->i == NULL || Pending->x == NULL
        || (is_matrix && Pending->j == NULL))
    { 
        // out of memory
        GB_Pending_free (&Pending) ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*PHandle) = Pending ;
    return (true) ;
}

