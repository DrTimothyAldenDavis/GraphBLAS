//------------------------------------------------------------------------------
// GB_queue_remove: remove a matrix from the matrix queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

bool GB_queue_remove            // remove matrix from queue
(
    GrB_Matrix A                // matrix to remove
)
{
    bool ok = true ;
    if (A->enqueued)
    { 
        #define GB_CRITICAL_SECTION                                         \
        {                                                                   \
            if (A->enqueued)                                                \
            {                                                               \
                GrB_Matrix Prev = (GrB_Matrix) (A->queue_prev) ;            \
                GrB_Matrix Next = (GrB_Matrix) (A->queue_next) ;            \
                if (Prev == NULL)                                           \
                {                                                           \
                    GB_Global_queue_head_set (Next) ;                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    Prev->queue_next = Next ;                               \
                }                                                           \
                if (Next != NULL)                                           \
                {                                                           \
                    Next->queue_prev = Prev ;                               \
                }                                                           \
                A->queue_prev = NULL ;                                      \
                A->queue_next = NULL ;                                      \
                A->enqueued = false ;                                       \
            }                                                               \
        }
        #include "GB_critical_section.c"
    }
    return (ok) ;
}

