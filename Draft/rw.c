
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// draft reader/writer solution

#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>

void writer (int k) ;
void reader (int k) ;

int work (int k)
{
    int x = 0 ;
    for (int i = 0 ; i < 100000 ; i++)
    {
        for (int j = 0 ; j < 10000 ; j++)
        {
            x++ ;
            if (x % 400 == k) x++ ;
        }
    }
    return (x) ;
}

omp_lock_t lock ;
int32_t reader_count = 0 ;
int32_t garbage = 0 ;       // approx count of active readers

void writer (int k)
{

    // lock the resource
    omp_set_lock (&lock) ;

    // solo writer here
    // do some useless writer work here
    {
        printf ("writer %d:%d starting %d\n", k, omp_get_thread_num ( ),
            garbage) ;
        if (garbage != 0) abort ( ) ;
        int x = work (k) ;
        printf ("writer %d:%d done%s\n", k, omp_get_thread_num ( ),
            (x == 0) ? "0" : ".")  ;
    }

    // unlock the resource
    omp_unset_lock (&lock) ;
}

void reader (int k)
{

    #pragma omp critical (reader)
    {
        reader_count++ ;
        if (reader_count == 1)
        {
            // first reader must get the lock
            omp_set_lock (&lock) ;
        }
    }

    // many readers here
    // do some useless reader work
    {
        #pragma omp atomic
        {
            garbage++ ;
        }
        int x = work (k) ;
        printf ("    reader %d:%d %s: %d\n", k, omp_get_thread_num ( ),
            (x == 0) ? "0" : ".", reader_count)  ;
        #pragma omp atomic
        {
            garbage-- ;
        }
    }

    #pragma omp critical (reader)
    {
        reader_count-- ;
        if (reader_count == 0)
        {
            // last reader must release the lock
            omp_unset_lock (&lock) ;
        }
    }
}

int main (void)
{
    omp_init_lock (&lock) ;
    reader_count = 0 ;
    printf ("# threads %d\n", omp_get_max_threads ( )) ;

    while (1)
    {
        #pragma omp parallel for num_threads(8) schedule(static,1)
        for (int k = 0 ; k < 100000 ; k++)
        {
            bool I_am_writer = (k % 17 == 0) ;
    //      printf ("thread %d:%d\n", k, omp_get_thread_num ( )) ;

            if (I_am_writer)
            {
                writer (k) ;
            }
            else
            {
                reader (k) ;
            }
            // sleep (1) ;
        }
    }
}
