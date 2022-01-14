
#include "GB.h"

GrB_Info GxB_Iterator_new (GxB_Iterator *iterator)
{
    GB_WHERE1 ("GxB_Iterator_new (&iterator)") ;
    GB_RETURN_IF_NULL (iterator) ;
    size_t header_size ;
    (*iterator) = GB_CALLOC (1, struct GB_Iterator_opaque, &header_size) ;
    if (*iterator == NULL)
    {
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*iterator)->header_size = header_size ;
    return (GrB_SUCCESS) ;
}

GrB_Info GxB_Iterator_free (GxB_Iterator *iterator)
{
    if (iterator != NULL)
    {
        size_t header_size = (*iterator)->header_size ;
        if (header_size > 0)
        {
            (*iterator)->header_size = 0 ;
            GB_FREE (iterator, header_size) ;
        }
    }
    return (GrB_SUCCESS) ;
}

