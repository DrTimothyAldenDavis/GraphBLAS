
#include <rmm_wrap.h>

int main()
{
    int size = 256; 

    RMM_Handle  *rmmH = (RMM_Handle *)malloc(size);
    rmm_create_handle( rmmH); 
    printf("RMM Handle created! \n");
	
    rmm_initialize( rmmH, (1<<10)*256, (1<<20)*256);
    printf("RMM initialized!\n");

    void *p;
    size_t buff_size = (1<<13)*252;
    p = rmm_allocate( &buff_size );
    rmm_deallocate( p, buff_size);

    rmm_destroy_handle(rmmH);

}
