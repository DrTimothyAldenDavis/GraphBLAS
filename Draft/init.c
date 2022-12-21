
    // optionally:
    rmm wrap init: gets all 6:
    cuda visible: [1 2 5 6 7 8]
    rmm_wrap_init_all ( ... on all visible GPUs ... )   // or ..
    GrB_init (GxB_NONBLOCKING_GPU) ;        /
    GrB_finalize ( ) ;                      // if we (GrB_init) did rmm_init, then rmm_fin here
    rmm_wrap_fin_all ( ... on all visible GPUs ... )    // or ..

