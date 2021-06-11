#include <stddef.h>
#include <stdio.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { rmm_host=0, rmm_host_pinned=1, rmm_device=2, rmm_managed=3 } RMM_MODE ;
typedef struct RMM_Handle RMM_Handle ; 

void rmm_create_handle( RMM_Handle **);
void rmm_destroy_handle(  RMM_Handle **);

void rmm_initialize( RMM_Handle *handle, RMM_MODE mode, size_t init_pool_size, size_t max_pool_size) ;

void *rmm_allocate(  size_t *size) ;
void rmm_deallocate( void *p, size_t size) ;

#ifdef __cplusplus
}
#endif
