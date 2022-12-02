
    // optionally:

    rmm wrap init: gets all 6:
    cuda visible: [1 2 5 6 7 8]

    rmm_wrap_init_all ( ... on all visible GPUs ... )   // or ..

    GrB_init (GxB_NONBLOCKING_GPU) ;        /
    GrB_finalize ( ) ;                      // if we (GrB_init) did rmm_init, then rmm_fin here

    rmm_wrap_fin_all ( ... on all visible GPUs ... )    // or ..



    #pragma omp threadprivate (GxB_Context_threadprivate)
    GxB_Context_threadprivate = NULL ;

    GxB_Context all = NULL ;
    GxB_Context_new (&all) ;        // also copied to GxB_Context_threadprivate
    GxB_Context_threadprivate = all ;

    GxB_Context_set (all, NTHREADS, 4) ;
    GxB_Context_set (all, GPU, always) ;
    GxB_Context_set (all, WHICHGPUS, [2 3]) ;
    GxB_Context_clear (all) ;       // set defaults
    GrB_mxm ( ...)
    GrB_reduce ( ...)
    GxB_Context_free (&all) ;

    #pragma omp parallel num_threads(4)
    for (k = 0 ; k < nthreads ; k++)
    {
        // thread k
        GxB_Context context = NULL ;
        GxB_Context_new (&context) ;        // also copied to GxB_Context_threadprivate

        // use the kth GPU
        GxB_Context_set (context, GPU, k) ; // k = 0,1,2,3,4,5
        ...
        GrB_mxm (... , desc) ;              // FIXME
        GrB_Matrix_dup (&C, A) ;            // no descriptor
        GrB_Matrix_build (C, ...) ;         // no descriptor
        GrB_Matrix_wait (C;                 // no descriptor

        C[k] = A[k]*B[k] on the kth gpu

        GrB_mxm (C [k], ... A [k], B [k], ..., desc)

        GxB_Context_free (&context) ;
    }


    (1) global setting:  which GPU(s), #threads, ...

    (2) if GxB_Context_threadprivate != NULL
            get it there instead

    (3) if GrB_Descriptor exists and is not GxB_DEFAULT
            get it there instead

rmm wrap init: gets all 6:
cuda visible: [1 2 5 6 7 8]

Context contents:

    OpenMP:
        nthreads_max
        chunk
                64K default
                work 256K "flops"
                at most 256K/64K = 4 threads at most

    GPU(s):
        ngpus
        list of GPUs to use, of size ngpus
            [ 1 2 4] to get device 2 5 7
        when to use the GPU
            always
            maybe (depending on problem size, location of data, ...)
            never
        chunk 64K

GrB_init time: (not part of the GxB_Context):
    
    find "all" the GPUs, or just the ones we want to use
    rmm_wrap_intialize:  startup rmm_managed_memory if not started already
    start 32 streams on each GPU

