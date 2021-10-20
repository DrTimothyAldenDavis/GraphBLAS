#include <gtest/gtest.h>
#include "../../rmm_wrap/rmm_wrap.h"

int main(int argc, char **argv) {

    size_t init_size, max_size;
    init_size = 256*(1ULL<<10);
    max_size  = 256*(1ULL<<20);

    rmm_wrap_create_handle();
    //printf(" pool init size %ld, max size %ld\n", init_size, max_size);
    rmm_wrap_initialize( rmm_wrap_managed, init_size, max_size );

    size_t buff_size = (1ULL<<13)+152;
    void *p = (void *)rmm_wrap_allocate( &buff_size );

    ::testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();

    rmm_wrap_deallocate( p, buff_size);
    rmm_wrap_destroy_handle();

    return r;
}