#include <stddef.h>
#include <stdio.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RMM_Handle RMM_Handle ; 

void rmm_create_handle( RMM_Handle *);
void rmm_destroy_handle(  RMM_Handle *);

void rmm_initialize( RMM_Handle *handle, size_t init_pool_size, size_t max_pool_size) ;

void *rmm_allocate(  size_t *size) ;
void rmm_deallocate( void *p, size_t size) ;

#ifdef __cplusplus
}
#endif
