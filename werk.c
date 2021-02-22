
// in Context: add the following:

    GB_void Werk [GB_WERK_SIZE] ;
    int pwerk = 0 ;

void *GB_werk_push
(
    // output
    bool *dynamically_allocated,
    // input
    bool do_calloc,
    size_t nitems,
    size_t size_of_item,
    GB_Context Context
)
{
    if (nitems == 0 || size_of_item == 0)
    {
        (*dynamically_allocated) = false ;
        return (NULL) ;
    }
    size_t size ;
    if (!GB_size_t_multiply (&size, nitems, size_of_item)) return (NULL) ;
    size_t freespace = GB_WERK_SIZE - Context->pwerk ;
    ASSERT (freespace % 8 == 0) ;
    if (size <= freespace)
    {
        // allocate werkspace from the stack
        (*dynamically_allocated) = false ;
        // round up to the nearest multiple of 8 bytes
        size = (size + 7) & (~0x7) ;
        GB_void *p = Context->Werk + Context->pwerk ;
        Context->pwerk += size ;
        if (do_calloc) memset (p, 0, size) ;
        return ((void *) p) ;
    }
    else
    {
        // allocate werkspace from malloc or calloc
        (*dynamically_allocated) = true ;
        return (do_calloc ?
            GB_calloc_memory (nitems, size_of_item) :
            GB_malloc_memory (nitems, size_of_item)) ;
    }
}

void *GB_werk_pop
(
    void *p,
    bool dynamically_allocated,
    size_t nitems,
    size_t size_of_item,
    GB_Context Context
)
{
    if (dynamically_allocated)
    {
        // werkspace was allocated from malloc/calloc
        GB_free_memory (p) ;
    }
    else
    {
        // werkspace was allocated from the Werk stack
        size_t size = nitems * size_of_item ;
        ASSERT (((GB_void *) p) + size == Context->Werk + Context->pwerk) ;
        Context->pwerk = ((GB_void *) p) - Context->Werk ;
    }
    return (NULL) ;
}

