//------------------------------------------------------------------------------
// GB_queue_insert:  insert a matrix at the head of the matrix queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

bool GB_queue_insert            // insert matrix at the head of queue
(
    GrB_Matrix A                // matrix to insert
)
{
    bool ok = true ;
    if ((A->Pending != NULL || A->nzombies > 0) && !(A->enqueued))
    { 
        #define GB_CRITICAL_SECTION                                         \
        {                                                                   \
            if ((A->Pending != NULL || A->nzombies > 0) && !(A->enqueued))  \
            {                                                               \
                GrB_Matrix Head = (GrB_Matrix) (GB_Global_queue_head_get ( )) ;\
                A->queue_next = Head ;                                      \
                A->queue_prev = NULL ;                                      \
                A->enqueued = true ;                                        \
                if (Head != NULL)                                           \
                {                                                           \
                    Head->queue_prev = A ;                                  \
                }                                                           \
                GB_Global_queue_head_set (A) ;                              \
            }                                                               \
        }
        #include "GB_critical_section.c"
    }
    return (ok) ;
}

