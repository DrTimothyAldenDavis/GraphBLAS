//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineHash_phase2_template:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine hash task, C(:,j)=A*B(:,j)
    //--------------------------------------------------------------------------

    // Given Hf [hash] split into (h,f)

    // h == 0  , f == 0 : unlocked and unoccupied.
    // h == i+1, f == 2 : unlocked, occupied by C(i,j).
    //                    Hx is initialized.
    // h == ..., f == 3 : locked.

    // 0 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    #ifdef GB_CHECK_MASK_ij
    #ifndef M_SIZE
    #define M_SIZE 1
    #endif
    const M_TYPE *GB_RESTRICT Mask = ((M_TYPE *) Mx) + (M_SIZE * pM_start) ;
    #endif

    if (team_size == 1)
    {

        //----------------------------------------------------------------------
        // single-threaded version
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i1 = i + 1 ;        // i1 = one-based index
                int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)

//#define GB_HASH_FUNCTION(i) ((((i) << 8) + (i)) & (hash_bits))
//#define GB_HASH(i) int64_t hash = GB_HASH_FUNCTION (i) ; ; GB_REHASH (hash,i)
// #define GB_REHASH(hash,i) hash = ((hash + 1) & (hash_bits))

int64_t hash = GB_HASH_FUNCTION (i) ;

                {
                    int64_t hf = Hf [hash] ;    // grab the entry
                    if (hf == i_unlocked)       // if true, update C(i,j)
                    { 
                        // hash entry occuppied by C(i,j): update it
                        GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
                        continue ;         // C(i,j) has been updated
                    }
                    if (hf == 0)
                    { 
                        // hash entry unoccuppied: fill it with C(i,j)
                        // Hx [hash] = t
                        GB_HX_WRITE (hash, t) ;
                        Hf [hash] = i_unlocked ; // unlock entry
                        continue ;
                    }
                    // otherwise: hash table occupied, but not with i
                }

                //for (GB_HASH (i))           // find i in hash table

                while (1)
                {
                    GB_REHASH (hash, i) ;
                    // hash++ ;
                    // hash &= hash_bits ;

                    int64_t hf = Hf [hash] ;    // grab the entry
                    if (hf == i_unlocked)       // if true, update C(i,j)
                    { 
                        // hash entry occuppied by C(i,j): update it
                        GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
                        break ;         // C(i,j) has been updated
                    }
                    if (hf == 0)
                    { 
                        // hash entry unoccuppied: fill it with C(i,j)
                        // Hx [hash] = t
                        GB_HX_WRITE (hash, t) ;
                        Hf [hash] = i_unlocked ; // unlock entry
                        break ;
                    }
                    // otherwise: hash table occupied, but not with i
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // multi-threaded version
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i1 = i + 1 ;        // i1 = one-based index
                int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                for (GB_HASH (i))           // find i in hash table
                {
                    int64_t hf ;
                    GB_ATOMIC_READ
                    hf = Hf [hash] ;        // grab the entry
                    #if GB_HAS_ATOMIC
                    if (hf == i_unlocked)  // if true, update C(i,j)
                    {
                        GB_ATOMIC_UPDATE_HX (hash, t) ;// Hx [.]+=t
                        break ;         // C(i,j) has been updated
                    }
                    #endif
                    int64_t h = (hf >> 2) ;
                    if (h == 0 || h == i1)
                    {
                        // h=0: unoccupied, h=i1: occupied by i
                        do  // lock the entry
                        {
                            // do this atomically:
                            // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                            GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3) ;
                        } while ((hf & 3) == 3) ; // owner: f=0 or 2
                        if (hf == 0) // f == 0
                        { 
                            // C(i,j) is a new entry in C(:,j)
                            // Hx [hash] = t
                            GB_ATOMIC_WRITE_HX (hash, t) ;
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }
                        if (hf == i_unlocked) // f == 2
                        { 
                            // C(i,j) already appears in C(:,j)
                            // Hx [hash] += t
                            GB_ATOMIC_UPDATE_HX (hash, t) ;
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }
                        // hash table occupied, but not with i
                        GB_ATOMIC_WRITE
                        Hf [hash] = hf ;  // unlock with prior value
                    }
                }
            }
        }
    }

    continue ;
}

#undef M_TYPE
#undef M_SIZE

