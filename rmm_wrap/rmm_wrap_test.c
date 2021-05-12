
#include <rmm_wrap.h>

int main()
{

    RMM_Handle  *rmmH;
	rmm_create_handle( rmmH); 
	printf("RMM Handle created! %ld\n", rmmH);
	
	rmm_initialize( rmmH, (1<<10)*256, (1<<20)*256);
	printf("RMM initialized!\n");

	void *p;
    size_t buff_size = (1<<13)*256;
	p = rmm_allocate(rmmH, buff_size );
	    rmm_deallocate(rmmH, p, buff_size);

}
