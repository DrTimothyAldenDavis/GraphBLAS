
    // establish the default global context (default GPUs: find them).  Either:
    LAGr_Init (GxB_NONBLOCKING_GPU,
        NULL, NULL, NULL, NULL,     // not needed: using Rapids malloc wrappers in GrB
        NULL) ;
    // and/or:
    GrB_init (GxB_NONBLOCKING_GPU) ;
    GxB_init (GxB_NONBLOCKING_GPU, NULL, NULL, NULL, NULL) ;


    //------------------------------------------------------------------------------
    // creating a context:
    //------------------------------------------------------------------------------

    #pragma omp threadprivate (GxB_Context_threadprivate)
    GxB_Context_threadprivate = NULL ;

    GxB_Context context = NULL ;
    GrB_mxm (... , NULL) ;              // use the global context

    GxB_Context_new (&context) ;
    // also copied to GxB_Context_threadprivate
    GxB_Context_make_me_private (context) ;
    // does: GxB_Context_threadprivate = context ;

    GxB_Context_get_my_private (&context) ;
    // does: context = GxB_Context_threadprivate = context ;

    GxB_Context_set (context, NTHREADS, 4) ;
    GxB_Context_set (context, GPU, always) ;
    GxB_Context_set (context, WHICHGPUS, [2 3]) ;
    GxB_Context_clear (context) ;       // set defaults

    GxB_set (desc1, context1) ;
    GxB_set (desc2, context1) ;
    GxB_set (desc3, context1) ;
    GxB_set (desc4, context1) ;
    GxB_set (desc5, context1) ;

    GxB_set (desc1, GxB_CONTEXT, context1) ;
    GxB_set (desc2, GxB_CONTEXT, context2) ;

    GxB_Context_make_me_private (context) ;
    GrB_mxm (... , NULL) ;              // use the threadprivate context
    GrB_reduce ( ...)

    GxB_Context_free (&context) ;

    GrB_mxm (C, M, accum, semiring, A, B, desc) ;
    // GrB_mxm (C, M, accum, semiring, A, B, f(desc,context)) ;

    GrB_set (A, GxB_CONTEXT, context) ;

    #pragma omp parallel num_threads(4)
    for (k = 0 ; k < nthreads ; k++)
    {
        // thread k
        GxB_Context context = NULL ;
        GxB_Context_new (&context) ;        // also copied to GxB_Context_threadprivate
        GxB_Context_threadprivate = context ;

        // use the kth GPU
        GxB_Context_set (context, GPU, k) ; // k = 0,1,2,3,4,5
        GxB_set (desc, GxB_CONTEXT, context) ;
        ...
        GrB_mxm (... , NULL) ;              // use the context
        GrB_mxm (... , desc) ;              // can use the context or descriptor

        // in MPI: move a matrix to another context
        GrB_set (A, GxB_CONTEXT, context1) ;
        // in MPI: move A to context2
        GrB_set (A, GxB_CONTEXT, context2) ;

        GrB_Matrix_dup (&C, A) ;            // no descriptor, use context of A

        GrB_set (C, GxB_CONTEXT, context) ;
        GrB_Matrix_build (C, ...) ;         // no descriptor

        GrB_Matrix_wait (C;                 // no descriptor

        C[k] = A[k]*B[k] on the kth gpu

        GrB_mxm (C [k], ... A [k], B [k], ..., desc)

        GrB_mxm (C, M, accum, semiring, A, B, desc)

        GxB_Context_free (&context) ;
    }

//------------------------------------------------------------------------------
precedence: from low to high:

    (1) global setting:  which GPU(s), #threads, ...
            use this if nothing else

    (2) if GxB_Context_threadprivate != NULL
            get it there instead

    (3) if GrB_Descriptor exists and is not GxB_DEFAULT
            get it there instead

    (4) if the C matrix has a context, use it
        but dup, sort uses A

        each GrB* method:  precedence
            mxm: use A, B, M, then C
            dup: use C
            sort: use input A, output P, output C

//------------------------------------------------------------------------------


// rmm wrap init: gets all 6:
cuda visible: [1 2 5 6 7 8]

Context contents:

    OpenMP:
        nthreads_max
        chunk
                64K default
                work 256K "flops"
                at most 256K/64K = 4 threads at most

    GPUs:
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
