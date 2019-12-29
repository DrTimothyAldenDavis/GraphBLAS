

#include "myhash.h"
#include "GB_sort.h"

// C = A*B using the hash method

int64_t myhash
(

    int64_t **Cp_handle,
    int64_t **Ci_handle,
    double  **Cx_handle,

    int64_t *restrict Ap,
    int64_t *restrict Ai,
    double  *restrict Ax,
    int64_t anrows,
    int64_t ancols,

    int64_t *restrict Bp,
    int64_t *restrict Bi,
    double  *restrict Bx,
    int64_t bnrows,
    int64_t bncols
)
{

    double tic [2] ;
    simple_tic (tic) ;

    //--------------------------------------------------------------------------
    // flop counts for each column of C
    //--------------------------------------------------------------------------

    int64_t cnrows = anrows ;
    int64_t cncols = bncols ;
    // int64_t *flops = mxCalloc (cncols, sizeof (int64_t)) ;

    int64_t flmax = 1 ;
    // double fltot = 0 ;

    for (int64_t j = 0 ; j < cncols ; j++)
    {
        int64_t fl = 0 ;
        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
        {
            // get B(k,j)
            int64_t k = Bi [pB] ;
            // flop count
            fl += (Ap [k+1] - Ap [k]) ;
        }
        // flops [j] = fl ;
        flmax = GB_IMAX (flmax, fl) ;
        // fltot += fl ;
    }

    // printf ("flmax %g\n", (double) flmax) ;

    //--------------------------------------------------------------------------
    // allocate hash table
    //--------------------------------------------------------------------------

    double hlog = log2 ((double) flmax) ;
    // printf ("hlog %g\n", hlog) ;
    int64_t hash_size = ((int64_t) 2) << ((int64_t) floor (hlog)) ;
    int64_t hash_bits = (hash_size-1) ;

    bool use_Gustavson = (hash_size >= cnrows) ;
    hash_size = GB_IMIN (hash_size, cnrows) ;

    // printf ("hash_size %g\n", (double) hash_size) ;

    int64_t *Hi = mxMalloc (hash_size * sizeof (int64_t)) ;

    //--------------------------------------------------------------------------
    // allocate Cp
    //--------------------------------------------------------------------------

    int64_t *Cp = mxMalloc ((cncols + 1) * sizeof (int64_t)) ;
    // int64_t ack = 0 ;

    //--------------------------------------------------------------------------
    // symbolic phase: count # of entries in each column of C
    //--------------------------------------------------------------------------

    for (int64_t j = 0 ; j < cncols ; j++)
    {

        //----------------------------------------------------------------------
        // clear the hash table
        //----------------------------------------------------------------------

        #if 1
        for (int64_t hash = 0 ; hash < hash_size ; hash++)
        {
            Hi [hash] = -1 ;
        }
        #else
        memset (Hi, 255, hash_size * sizeof (int64_t)) ;
        #endif

        //----------------------------------------------------------------------
        // count # of entries in C(:,j)
        //----------------------------------------------------------------------

        int64_t cjnz = 0 ;
        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
        {
            // get B(k,j)
            int64_t k = Bi [pB] ;
            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
            {
                // get A(i,k)
                int64_t i = Ai [pA] ;

                // find A(i,k) in the hash table
                int64_t hash = (i * 107) & (hash_bits) ;

                // int64_t ok = 0 ;
                while (1)
                {
                    // ack++ ;
                    int64_t h = Hi [hash] ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hi [hash]
                        break ;
                    }
                    else if (h == -1)
                    {
                        // Hi [hash] is empty, so i is not in the hash table.
                        // add i to the hash table at this location
                        Hi [hash] = i ;
                        cjnz++ ;
                        break ;
                    }
                    else
                    {
                        // linear probing
                        hash = (hash + 1) & (hash_bits) ;
                    }
                    // ok ++ ;
                    // if (ok > cnrows) mexErrMsgTxt ("hey!\n") ;
                }
            }
        }

        Cp [j] = cjnz ;

        // printf ("nz in C(:,%g): %g\n", (double) j, (double) cjnz) ;
    }

    //--------------------------------------------------------------------------
    // cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, cncols, NULL, 1) ;
    int64_t cnz = Cp [cncols] ;

    double tsym = simple_toc (tic) ;
    // printf ("cnz %g, hash overhead %g\n", (double) cnz,
        // ((double) ack) / (double) fltot) ;
    printf ("symbolic time %g\n", tsym) ;
    simple_tic (tic) ;

    //--------------------------------------------------------------------------
    // allocate Ci and Cx
    //--------------------------------------------------------------------------

    int64_t *Ci = mxMalloc (GB_IMAX (cnz, 1) * sizeof (int64_t)) ;
    double  *Cx = mxMalloc (GB_IMAX (cnz, 1) * sizeof (double )) ;
    double  *Hx = mxMalloc (hash_size * sizeof (double )) ;

    //--------------------------------------------------------------------------
    // numeric phase
    //--------------------------------------------------------------------------

    int64_t pC = 0 ;

    for (int64_t j = 0 ; j < cncols ; j++)
    {

        //----------------------------------------------------------------------
        // clear the hash table
        //----------------------------------------------------------------------

        #if 1
        for (int64_t hash = 0 ; hash < hash_size ; hash++)
        {
            Hi [hash] = -1 ;
        }
        #else
        memset (Hi, 255, hash_size * sizeof (int64_t)) ;
        #endif

        //----------------------------------------------------------------------
        // compute C(:,j)
        //----------------------------------------------------------------------

        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
        {
            // get B(k,j)
            int64_t k  = Bi [pB] ;
            double bkj = Bx [pB] ;

            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
            {
                // get A(i,k)
                int64_t i  = Ai [pA] ;
                double t = Ax [pA] * bkj ;

                // find A(i,k) in the hash table
                int64_t hash = (i * 107) & (hash_bits) ;

                while (1)
                {
                    int64_t h = Hi [hash] ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hi [hash]
                        Hx [hash] += t ;
                        break ;
                    }
                    else if (h == -1)
                    {
                        // Hi [hash] is empty, so i is not in the hash table.
                        // add i to the hash table at this location
                        Hi [hash] = i ;
                        Hx [hash] = t ;
                        break ;
                    }
                    else
                    {
                        // linear probing
                        hash = (hash + 1) & (hash_bits) ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // copy C(:,j) from the hash table into C
        //----------------------------------------------------------------------

        for (int64_t hash = 0 ; hash < hash_size ; hash++)
        {
            int64_t i = Hi [hash] ;
            if (i >= 0)
            {
                Ci [pC] = i ;
                #if 1
                Cx [pC] = Hx [hash] ;
                #endif
                pC++ ;
            }
        }

        //----------------------------------------------------------------------
        // sort C(:,j)
        //----------------------------------------------------------------------

        #if 1
        qsort_1b_double (Ci + Cp [j], (void *) (Cx + Cp [j]), sizeof (double),
            Cp [j+1] - Cp [j]) ;
        #else

        // this is just barely slower:
        GB_qsort_1a (Ci + Cp [j], Cp [j+1] - Cp [j]) ;

        for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            int64_t i = Ci [p] ;

                // find C(i,j) in the hash table
                int64_t hash = (i * 107) & (hash_bits) ;

                while (1)
                {
                    int64_t h = Hi [hash] ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hi [hash]
                        Cx [p] = Hx [hash] ;
                        break ;
                    }
                    #if 0
                    else if (h == -1)
                    {
                        // Hi [hash] is empty, so i is not in the hash table.
                        // add i to the hash table at this location
                        mexErrMsgTxt ("ack!") ;
                        break ;
                    }
                    #endif
                    else
                    {
                        // linear probing
                        hash = (hash + 1) & (hash_bits) ;
                    }
                }

        }
        #endif

    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*Cp_handle) = Cp ;
    (*Ci_handle) = Ci ;
    (*Cx_handle) = Cx ;

    mxFree (Hi) ;
    mxFree (Hx) ;
    double tnum = simple_toc (tic) ;
    double ttot = tsym + tnum ;
    printf ("numeric time %g\n", tnum) ;
    printf ("total %g\n", ttot) ;
    return (hash_size) ;
}

