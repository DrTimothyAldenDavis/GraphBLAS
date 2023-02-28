{

    //----------------------------------------------------------------------
    // Method30: C, A, B are all full
    //----------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    { 
        // C (i,j) = A (i,j) + B (i,j)
        GB_LOAD_A (aij, Ax, p, A_iso) ;
        GB_LOAD_B (bij, Bx, p, B_iso) ;
        GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
    }
}

