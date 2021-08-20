
#include "rmm_wrap.h"

int main()
{

    RMM_Handle  *rmmH ;
    rmm_create_handle( &rmmH); 
    //printf("RMM Handle created! \n");

    size_t init_size, max_size;      
    init_size = 256*(1ULL<<10);
    max_size  = 256*(1ULL<<20);
	
    //printf(" pool init size %ld, max size %ld\n", init_size, max_size);
    rmm_initialize( rmmH, rmm_managed, init_size, max_size );
    printf("RMM initialized!  in managed mode\n");

    void *p;
    size_t buff_size = (1ULL<<13)+152;

    printf(" asked for %ld", buff_size);
    fflush(stdout);
    p = rmm_allocate( &buff_size );
    printf(" actually allocated  %ld\n", buff_size);
    fflush(stdout);
    rmm_deallocate( p, buff_size);


    rmm_initialize( rmmH, rmm_device, init_size, max_size );
    printf("RMM initialized!  in device mode\n");

    buff_size = (1ULL<<13)+157;
    printf(" asked for %ld", buff_size);
    fflush(stdout);
    p = rmm_allocate( &buff_size );
    printf(" actually allocated  %ld\n", buff_size);
    fflush(stdout);
    rmm_deallocate( p, buff_size);

    rmm_destroy_handle( &rmmH);


}
