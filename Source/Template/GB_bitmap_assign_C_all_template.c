
            // iterate over all of C(:,:)
            int64_t pC ;
            int nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (pC = 0 ; pC < cnzmax ; pC++)
            { 
                int64_t iC = pC % cvlen ;
                int64_t jC = pC / cvlen ;
                GB_GET_MIJ (mij, pC) ;          // mij = Mask (pC)
                GB_CIJ_WORK (mij, pC) ;         // operate on C(iC,jC)
            }

