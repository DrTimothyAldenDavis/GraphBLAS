//------------------------------------------------------------------------------
// GrB_finalize: finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GrB_finalize must be called as the last GraphBLAS function, per the
// GraphBLAS C API Specification.  

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

GrB_Info GrB_finalize ( )
{ 

    //--------------------------------------------------------------------------
    // free all workspace
    //--------------------------------------------------------------------------

    for (int Sauna_id = 0 ; Sauna_id < GxB_NTHREADS_MAX ; Sauna_id++)
    {
        GB_Sauna_free (Sauna_id) ;
    }

    //--------------------------------------------------------------------------
    // destroy the queue
    //--------------------------------------------------------------------------

    GB_CRITICAL (GB_queue_destroy ( )) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

