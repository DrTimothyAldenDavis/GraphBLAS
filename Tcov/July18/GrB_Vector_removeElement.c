//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Removes a single entry, V (i), from the vector V.

#include "GB.h"

#define GB_FREE_ALL ;
#define GB_WHERE_STRING "GrB_Vector_removeElement (v, i)"

//------------------------------------------------------------------------------
// GB_removeElement: remove V(i) if it exists
//------------------------------------------------------------------------------

static inline bool GB_removeElement
(
    GrB_Vector V,
    GrB_Index i
)
{

    //--------------------------------------------------------------------------
    // get the pattern of the vector
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Vp = V->p ;
    const int64_t *GB_RESTRICT Vi = V->i ;
    bool found ;

    // remove from a GrB_Vector
    int64_t pleft = 0 ;
    int64_t pright = Vp [1] - 1 ;

    //--------------------------------------------------------------------------
    // binary search in kth vector for index i
    //--------------------------------------------------------------------------

    // Time taken for this step is at most O(log(nnz(V))).
    bool is_zombie ;
    int64_t nzombies = V->nzombies ;
    GB_BINARY_SEARCH_ZOMBIE (i, Vi, pleft, pright, found, nzombies, is_zombie) ;

    //--------------------------------------------------------------------------
    // remove the entry
    //--------------------------------------------------------------------------

    if (found && !is_zombie)
    {   GB_cov[3593]++ ;
// covered (3593): 57
        // V(i) becomes a zombie
        V->i [pleft] = GB_FLIP (i) ;        // ok: V is sparse
        V->nzombies++ ;
    }

    return (found) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_removeElement
(
    GrB_Vector V,               // vector to remove entry from
    GrB_Index i                 // index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (V) ;

    // GB_ENSURE_SPARSE (V) ;
    if (GB_IS_FULL (V))
    {   GB_cov[3594]++ ;
// NOT COVERED (3594):
        // convert V from full to sparse
        GB_WHERE (V, GB_WHERE_STRING) ;
        GrB_Info info = GB_full_to_sparse ((GrB_Matrix) V, Context) ;
        if (info != GrB_SUCCESS)
        {   GB_cov[3595]++ ;
// NOT COVERED (3595):
            return (info) ;
        }
    }

    // check index
    if (i >= V->vlen)
    {   GB_cov[3596]++ ;
// covered (3596): 2
        GB_WHERE (V, GB_WHERE_STRING) ;
        GB_ERROR (GrB_INVALID_INDEX, "Row index "
            GBu " out of range; must be < " GBd, i, V->vlen) ;
    }

    bool V_is_pending = GB_PENDING (V) ; 
    if (V->nzmax == 0 && !V_is_pending)
    {   GB_cov[3597]++ ;
// covered (3597): 2
        // quick return
        return (GrB_SUCCESS) ;
    }

    // remove the entry
    if (GB_removeElement (V, i))
    {   GB_cov[3598]++ ;
// covered (3598): 57
        // found it; no need to assemble pending tuples
        return (GrB_SUCCESS) ;
    }

    // assemble any pending tuples; zombies are OK
    if (V_is_pending)
    {   GB_cov[3599]++ ;
// covered (3599): 24
        GrB_Info info ;
        GB_WHERE (V, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Vector_removeElement") ;
        GB_OK (GB_Matrix_wait ((GrB_Matrix) V, Context)) ;
        ASSERT (!GB_ZOMBIES (V)) ;
        ASSERT (!GB_PENDING (V)) ;
        GB_BURBLE_END ;
    }

    // look again; remove the entry if it was a pending tuple
    GB_removeElement (V, i) ;
    return (GrB_SUCCESS) ;
}

