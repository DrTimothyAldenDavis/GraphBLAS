

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
    int64_t bncols,

    int64_t *nonempty_result
)
{

    //--------------------------------------------------------------------------
    // flop counts for each column of C
    //--------------------------------------------------------------------------

    int64_t cnrows = anrows ;
    int64_t cncols = bncols ;

    int64_t flmax = 1 ;

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
        flmax = GB_IMAX (flmax, fl) ;
    }

    //--------------------------------------------------------------------------
    // allocate the hash table
    //--------------------------------------------------------------------------

    // If Gustavson's method is used:
    //
    //      hash_size is cnrows.
    //      Hi is not allocated.
    //      Hf and Hx are both of size hash_size.
    //
    //      (Hf [i] == mark) is true if i is in the hash table.
    //      Hx [i] is the value of C(i,j) during the numeric phase.
    //
    //      Gustavson's method is used if the hash_size for the Hash method
    //      is greater than or equal to cnrows/4.
    //
    // If the Hash method is used:
    //
    //      hash_size is 2 times the smallest power of 2 that is larger than
    //      the # of flops required for any column C(:,j) being computed.  This
    //      ensures that all entries have space in the hash table, and that the
    //      hash occupancy will never be more than 50%.  It is always smaller
    //      than cnrows/4 (otherwise, Gustavson's method is used).
    //
    //      A hash function is used for the ith entry:
    //          hash = (i * 107) % hash_size
    //      If a collision occurs, linear probing is used.
    //
    //      (Hf [hash] == mark) is true if the position is occupied.
    //      (Hi [hash] == i) gives the row index i that occupies
    //      that position.
    //      Hx [hash] is the value of C(i,j) during the numeric phase.
    //
    // For both methods:
    //
    //      Hf starts out all zero (via calloc), and mark starts out as 1.
    //      To clear all of Hf, mark is incremented, so that all entries in
    //      Hf are not equal to mark.

    double hlog = log2 ((double) flmax) ;
    int64_t hash_size = ((int64_t) 2) << ((int64_t) floor (hlog) + 1) ;

    int64_t *restrict Hi = NULL ;

    bool use_Gustavson = (hash_size >= cnrows/4) ;
    if (use_Gustavson)
    {
        hash_size = cnrows ;
    }
    else
    {
        Hi = mxMalloc (hash_size * sizeof (int64_t)) ;
    }

    int64_t *restrict Hf = mxCalloc (hash_size, sizeof (int64_t)) ;
    int64_t mark = 1 ;
    int64_t hash_bits = (hash_size-1) ;

    //--------------------------------------------------------------------------
    // allocate Cp
    //--------------------------------------------------------------------------

    int64_t *restrict Cp = mxMalloc ((cncols + 1) * sizeof (int64_t)) ;

    //--------------------------------------------------------------------------
    // symbolic phase: count # of entries in each column of C
    //--------------------------------------------------------------------------

    if (use_Gustavson)
    {

        //----------------------------------------------------------------------
        // Gustavon's method
        //----------------------------------------------------------------------

        for (int64_t j = 0 ; j < cncols ; j++)
        {

            //------------------------------------------------------------------
            // count the entries in C(:,j)
            //------------------------------------------------------------------

            // TODO: if nnz (B (:,j)) is 1, then nnz (C (:,j)) == nnz (A (:,k))

            mark++ ;
            int64_t cjnz = 0 ;
            for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
            {
                // get B(k,j)
                int64_t k = Bi [pB] ;
                for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k)
                    //----------------------------------------------------------

                    int64_t i = Ai [pA] ;

                    //----------------------------------------------------------
                    // find A(i,k) in the hash table
                    //----------------------------------------------------------

                    if (Hf [i] != mark)
                    {
                        // hash entry is not occuppied;
                        // add i to the hash table at this location
                        Hf [i] = mark ;
                        cjnz++ ;
                    }
                }
            }
            Cp [j] = cjnz ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // hash method
        //----------------------------------------------------------------------

        for (int64_t j = 0 ; j < cncols ; j++)
        {

            //------------------------------------------------------------------
            // count the entries in C(:,j)
            //------------------------------------------------------------------

            // TODO: if nnz (B (:,j)) is 1, then nnz (C (:,j)) == nnz (A (:,k))

            mark++ ;
            int64_t cjnz = 0 ;
            for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
            {
                // get B(k,j)
                int64_t k = Bi [pB] ;
                for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k)
                    //----------------------------------------------------------

                    int64_t i = Ai [pA] ;

                    //----------------------------------------------------------
                    // find A(i,k) in the hash table
                    //----------------------------------------------------------

                    int64_t hash = (i * 107) & (hash_bits) ;
                    while (1)
                    {
                        if (Hf [hash] == mark)
                        {
                            // hash entry is occuppied
                            int64_t h = Hi [hash] ;
                            if (h == i)
                            {
                                // i already in the hash table, at Hi [hash]
                                break ;
                            }
                            else
                            {
                                // linear probing
                                hash = (hash + 1) & (hash_bits) ;
                            }
                        }
                        else
                        {
                            // hash entry is not occuppied;
                            // add i to the hash table at this location
                            Hf [hash] = mark ;
                            Hi [hash] = i ;
                            cjnz++ ;
                            break ;
                        }
                    }
                }
            }

            Cp [j] = cjnz ;
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, cncols, nonempty_result, 1) ;
    int64_t cnz = Cp [cncols] ;

    //--------------------------------------------------------------------------
    // allocate Ci and Cx
    //--------------------------------------------------------------------------

    int64_t *restrict Ci = mxMalloc (GB_IMAX (cnz, 1) * sizeof (int64_t)) ;
    double  *restrict Cx = mxMalloc (GB_IMAX (cnz, 1) * sizeof (double )) ;
    double  *restrict Hx = mxMalloc (hash_size * sizeof (double )) ;

    //--------------------------------------------------------------------------
    // numeric phase
    //--------------------------------------------------------------------------

    if (use_Gustavson)
    {

        //----------------------------------------------------------------------
        // Gustavson's method
        //----------------------------------------------------------------------

        for (int64_t j = 0 ; j < cncols ; j++)
        {

            //------------------------------------------------------------------
            // compute the pattern and values of C(:,j)
            //------------------------------------------------------------------

            int64_t pC = Cp [j] ;
            int64_t cjnz = Cp [j+1] - pC ;

            if (cjnz == cnrows)
            {

                //--------------------------------------------------------------
                // C(:,j) is dense; compute directly in Ci and Cx
                //--------------------------------------------------------------

                for (int64_t p = pC, i = 0 ; p < Cp [j+1] ; p++, i++)
                {
                    Ci [p] = i ;
                    Cx [p] = 0 ;
                }

                for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                {
                    // get B(k,j)
                    int64_t k  = Bi [pB] ;
                    double bkj = Bx [pB] ;

                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i  = Ai [pA] ;
                        double aik = Ax [pA] ;
                        Cx [pC + i] += aik * bkj ;
                    }
                }

            }
            else
            {

                // TODO: add case for nnz (B (:,j)) == 1

                //--------------------------------------------------------------
                // C(:,j) is sparse; compute in hash table
                //--------------------------------------------------------------

                mark++ ;
                for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                {
                    // get B(k,j)
                    int64_t k  = Bi [pB] ;
                    double bkj = Bx [pB] ;

                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i  = Ai [pA] ;
                        double aik = Ax [pA] ;
                        double t = aik * bkj ;

                        if (Hf [i] != mark)
                        {
                            Hf [i] = mark ;
                            Hx [i] = t ;
                            Ci [pC++] = i ;
                        }
                        else
                        {
                            Hx [i] += t ;
                        }
                    }
                }

                //--------------------------------------------------------------
                // sort the pattern of C(:,j)
                //--------------------------------------------------------------

                GB_qsort_1a (Ci + Cp [j], cjnz) ;

                //--------------------------------------------------------------
                // gather the values into C(:,j)
                //--------------------------------------------------------------

                for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
                {
                    int64_t i = Ci [p] ;
                    Cx [p] = Hx [i] ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // hash method
        //----------------------------------------------------------------------

        for (int64_t j = 0 ; j < cncols ; j++)
        {

            //------------------------------------------------------------------
            // compute the pattern and values of C(:,j)
            //------------------------------------------------------------------

                // TODO: add case for nnz (B (:,j)) == 1

            int64_t pC = Cp [j] ;
            mark++ ;
            for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
            {
                // get B(k,j)
                int64_t k  = Bi [pB] ;
                double bkj = Bx [pB] ;

                for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                {
                    // get A(i,k)
                    int64_t i  = Ai [pA] ;
                    double aik = Ax [pA] ;
                    double t = aik * bkj ;

                    //----------------------------------------------------------
                    // find A(i,k) in the hash table
                    //----------------------------------------------------------

                    int64_t hash = (i * 107) & (hash_bits) ;
                    while (1)
                    {
                        if (Hf [hash] == mark)
                        {
                            // hash entry is occuppied
                            int64_t h = Hi [hash] ;
                            if (h == i)
                            {
                                // i already in the hash table, at Hi [hash]
                                Hx [hash] += t ;
                                break ;
                            }
                            else
                            {
                                // linear probing
                                hash = (hash + 1) & (hash_bits) ;
                            }
                        }
                        else
                        {
                            // hash entry is not occuppied;
                            // add i to the hash table at this location
                            Hf [hash] = mark ;
                            Hi [hash] = i ;
                            Hx [hash] = t ;
                            Ci [pC++] = i ;
                            break ;
                        }
                    }
                }
            }

            //------------------------------------------------------------------
            // sort the pattern of C(:,j)
            //------------------------------------------------------------------

            GB_qsort_1a (Ci + Cp [j], Cp [j+1] - Cp [j]) ;

            //------------------------------------------------------------------
            // gather the values of C(:,j)
            //------------------------------------------------------------------

            for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
            {
                int64_t i = Ci [p] ;

                //--------------------------------------------------------------
                // find C(i,j) in the hash table
                //--------------------------------------------------------------

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
                    else
                    {
                        // linear probing
                        hash = (hash + 1) & (hash_bits) ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*Cp_handle) = Cp ;
    (*Ci_handle) = Ci ;
    (*Cx_handle) = Cx ;

    if (Hi != NULL) mxFree (Hi) ;
    mxFree (Hx) ;
    mxFree (Hf) ;
    return (hash_size) ;
}

