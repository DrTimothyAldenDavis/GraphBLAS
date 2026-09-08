//------------------------------------------------------------------------------
// CUDA/rmm/GB_rmm_wrap.h: include file for GB_rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef RMM_WRAP_H
#define RMM_WRAP_H

#ifdef __cplusplus
extern "C" {
#endif

#if 0
typedef enum
{
    rmm_wrap_host = 0,
    rmm_wrap_host_pinned = 1,
    rmm_wrap_device = 2,
    rmm_wrap_managed = 3
}
RMM_MODE ;
#endif

// determine if RMM has been initialized
bool rmm_wrap_is_initialized (void) ;

// create an RMM resource
int rmm_wrap_initialize
(
    int device_id
//  , RMM_MODE mode,
//  size_t init_pool_memsize,
//  size_t max_pool_memsize
) ;

// initialize rmm_wrap_contexts for each device in CUDA_VISIBLE_DEVICES
// (or single device_id 0 if not specified)
int rmm_wrap_initialize_all_same
(
    void
//  RMM_MODE mode,
//  size_t init_pool_memsize,
//  size_t max_pool_memsize
) ;

// destroy an RMM resource
void rmm_wrap_finalize (void) ;

// allocate/deallocate methods
void *rmm_allocate   (int device_id, size_t size) ;
void  rmm_deallocate (int device_id, void *p) ;

#ifdef GRAPHBLAS_HAS_CUDA

    // create a pair of malloc/free methods for a given device id
    #define GB_RMM_MALLOC_FREE(id)                  \
        void *GB_rmm_malloc_ ## id (size_t size)    \
        {                                           \
            return (rmm_allocate (id, size)) ;      \
        }                                           \
        void GB_rmm_free_ ## id (void *p)           \
        {                                           \
            rmm_deallocate (id, p) ;                \
        }

#else

    // no CUDA devices available; create stubs that do nothing
    #define GB_RMM_MALLOC_FREE(id)                  \
        void *GB_rmm_malloc_ ## id (size_t size)    \
        {                                           \
            /* do not allocate anything */          \
            printf ("no malloc for device %d\n", id) ; \
            return (NULL) ;                         \
        }                                           \
        void GB_rmm_free_ ## id (void *p)           \
        {                                           \
            printf ("no free for device %d\n", id) ; \
            /* do nothing */ ;                      \
        }

#endif

#define GB_RMM_MALLOC_FREE_DECLARE(id)              \
        void *GB_rmm_malloc_ ## id (size_t size) ;  \
        void GB_rmm_free_ ## id (void *p) ;

GB_RMM_MALLOC_FREE_DECLARE (0) ;
GB_RMM_MALLOC_FREE_DECLARE (1) ;
GB_RMM_MALLOC_FREE_DECLARE (2) ;
GB_RMM_MALLOC_FREE_DECLARE (3) ;
GB_RMM_MALLOC_FREE_DECLARE (4) ;
GB_RMM_MALLOC_FREE_DECLARE (6) ;
GB_RMM_MALLOC_FREE_DECLARE (7) ;
GB_RMM_MALLOC_FREE_DECLARE (8) ;
GB_RMM_MALLOC_FREE_DECLARE (9) ;

GB_RMM_MALLOC_FREE_DECLARE (10) ;
GB_RMM_MALLOC_FREE_DECLARE (11) ;
GB_RMM_MALLOC_FREE_DECLARE (12) ;
GB_RMM_MALLOC_FREE_DECLARE (13) ;
GB_RMM_MALLOC_FREE_DECLARE (14) ;
GB_RMM_MALLOC_FREE_DECLARE (16) ;
GB_RMM_MALLOC_FREE_DECLARE (17) ;
GB_RMM_MALLOC_FREE_DECLARE (18) ;
GB_RMM_MALLOC_FREE_DECLARE (19) ;

GB_RMM_MALLOC_FREE_DECLARE (20) ;
GB_RMM_MALLOC_FREE_DECLARE (21) ;
GB_RMM_MALLOC_FREE_DECLARE (22) ;
GB_RMM_MALLOC_FREE_DECLARE (23) ;
GB_RMM_MALLOC_FREE_DECLARE (24) ;
GB_RMM_MALLOC_FREE_DECLARE (26) ;
GB_RMM_MALLOC_FREE_DECLARE (27) ;
GB_RMM_MALLOC_FREE_DECLARE (28) ;
GB_RMM_MALLOC_FREE_DECLARE (29) ;

GB_RMM_MALLOC_FREE_DECLARE (30) ;
GB_RMM_MALLOC_FREE_DECLARE (31) ;
GB_RMM_MALLOC_FREE_DECLARE (32) ;
GB_RMM_MALLOC_FREE_DECLARE (33) ;
GB_RMM_MALLOC_FREE_DECLARE (34) ;
GB_RMM_MALLOC_FREE_DECLARE (36) ;
GB_RMM_MALLOC_FREE_DECLARE (37) ;
GB_RMM_MALLOC_FREE_DECLARE (38) ;
GB_RMM_MALLOC_FREE_DECLARE (39) ;

GB_RMM_MALLOC_FREE_DECLARE (40) ;
GB_RMM_MALLOC_FREE_DECLARE (41) ;
GB_RMM_MALLOC_FREE_DECLARE (42) ;
GB_RMM_MALLOC_FREE_DECLARE (43) ;
GB_RMM_MALLOC_FREE_DECLARE (44) ;
GB_RMM_MALLOC_FREE_DECLARE (46) ;
GB_RMM_MALLOC_FREE_DECLARE (47) ;
GB_RMM_MALLOC_FREE_DECLARE (48) ;
GB_RMM_MALLOC_FREE_DECLARE (49) ;

GB_RMM_MALLOC_FREE_DECLARE (50) ;
GB_RMM_MALLOC_FREE_DECLARE (51) ;
GB_RMM_MALLOC_FREE_DECLARE (52) ;
GB_RMM_MALLOC_FREE_DECLARE (53) ;
GB_RMM_MALLOC_FREE_DECLARE (54) ;
GB_RMM_MALLOC_FREE_DECLARE (56) ;
GB_RMM_MALLOC_FREE_DECLARE (57) ;
GB_RMM_MALLOC_FREE_DECLARE (58) ;
GB_RMM_MALLOC_FREE_DECLARE (59) ;

GB_RMM_MALLOC_FREE_DECLARE (60) ;
GB_RMM_MALLOC_FREE_DECLARE (61) ;
GB_RMM_MALLOC_FREE_DECLARE (62) ;
GB_RMM_MALLOC_FREE_DECLARE (63) ;

#ifdef __cplusplus
}
#endif
#endif

