
            // iterate over all of C(iC,:)
            int64_t iC = I [0] ;
            int64_t jC ;
            int nthreads = GB_nthreads (cvdim, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (jC = 0 ; jC < cvdim ; jC++)
            {
                int64_t pC = iC + jC * cvlen ;
                GB_GET_MIJ (mij, jC) ;          // mij = Mask (jC)
                GB_CIJ_WORK (mij, pC) ;         // operate on C(iC,jC)
            }

