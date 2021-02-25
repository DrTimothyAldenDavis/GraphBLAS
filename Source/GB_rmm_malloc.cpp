
#include "GB.h"

void *GB_rmm_malloc (size_t size)
{
    void *rmm_pool = GB_Global_rmm_pool_get ( ) ;
    ASSERT (rmm_pool != NULL) ;

    p = ...  // ...  C++ magic

    return (p) ;
}

