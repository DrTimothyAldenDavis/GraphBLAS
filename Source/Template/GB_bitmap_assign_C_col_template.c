
            // iterate over all of C(:,jC)
            int64_t iC ;
            int64_t jC = J [0] ;
            int nthreads = GB_nthreads (cvlen, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (iC = 0 ; iC < cvlen ; iC++)
            {
                int64_t pC = iC + jC * cvlen ;
                GB_GET_MIJ (mij, iC) ;          // mij = Mask (iC)
                GB_CIJ_WORK (mij, pC) ;         // operate on C(iC,jC)
            }

