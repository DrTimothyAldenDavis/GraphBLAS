//************************************************************************
//
// PURPOSE:  This is a test-bed for exploring different sort algorithms.
//    The program generates an array of pseudo-random values, sorts the array
//    and then tests the results.  The datatype of the elements of the array
//    is any of the standard integer types in C as specified by the macro, Int
//
// Usage: to run test on arrays of size 234, run the program as:
//
//       ./testSort 234
//
//    Uncomment the VERBOSE definition if you want to explicitly output 
//    the initial array, the sorted arrays, and specific points where
//    errors occur. 
//
//************************************************************************

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>

#include <omp.h>

// sort function prototypes and the type "Int" which is set
// to one of the standard C integer data types
#include "sort.h"

#define NDEFAULT  1024   
// #define VERBOSE 1
#define PAR1
// #define PAR2
   
// Internal utility functions
void debug_output(long N, int64_t checksum, Int *a);
long TestResults(long N, int64_t checksumOrig, int64_t *checksumSorted, Int *a);
void copyArr(Int *dst, Int *src, long N);
void clrArr(Int *arr, long N);

int main(int argc, char **argv)
{
// amaster is the initial array a clean (unsorted) copy is saved.
// We copy amaster into a before sorting.  
   Int *amaster, *a, *work;
   long N=0,err;
   int64_t checksumOrig, checksumSorted ;
   double timepoint;

   if (argc == 2) 
     N = atol(argv[1]);
   else {
     N = NDEFAULT;
   }

   if (N<8)N=8;

   amaster = (Int *) malloc(N * sizeof(Int));
   a       = (Int *) malloc(N * sizeof(Int));
   work    = (Int *) malloc(N * sizeof(Int));
   if((a == NULL) || (work == NULL)){
      printf("memory allocation error for arrays of size %ld\n",N);
      exit(-1);
   }

//
// set static mode so the number of threads won't change
// unless explicitly requested. Fill the array, amaster, with 
// random values
//
   omp_set_dynamic(0);
   checksumOrig = 0;
   #pragma omp parallel firstprivate(N) shared(checksumOrig, amaster)
   {
      #pragma omp single
        printf("%d thrds to sort an array of %ld numbers\n",omp_get_num_threads(),N);

      #pragma omp for reduction(+:checksumOrig)
      for (int i=0;i<N;i++) 
      {
//          amaster[i] = (Int) rand()%N;
          amaster[i] = (Int)(N-1-i);
          checksumOrig += amaster[i];
      }
   }
    
#ifdef VERBOSE 
   debug_output(N, checksumOrig, amaster);
#endif    

//
// Serial Sort of the array
//
   copyArr(a, amaster, N);
   clrArr(work, N);
   timepoint = omp_get_wtime();
   ssmergesort(a, work, N);
   printf(" sorting complete in %f seconds\n",(float)(omp_get_wtime() - timepoint));

   err = TestResults(N, checksumOrig, &checksumSorted, a);

   if(err>0)printf("Errors in sort: %ld\n",err);

#ifdef VERBOSE
   debug_output(N, checksumSorted, a);
#endif    

//
// Parallel Sort of the array .. 4-way splitting
//
#ifdef PAR1
   copyArr(a, amaster, N);
   clrArr(work, N);
   timepoint = omp_get_wtime();
   parsort1(a, work, N);
   printf(" par 1 sorting complete in %g seconds\n",(double)(omp_get_wtime() - timepoint));

   err = TestResults(N, checksumOrig, &checksumSorted, a);

   if(err>0)printf("%ld Errors in parsort1.\n",err);

#ifdef VERBOSE
  debug_output(N, checksumSorted, a);
#endif    
#endif
//
// Parallel Sort of the array ... binary splitting
//
#ifdef PAR2
   copyArr(a, amaster, N);
   clrArr(work, N);
   timepoint = omp_get_wtime();
   parsort2(a, work, N);
   printf(" par 2 sorting complete in %g seconds\n",(double)(omp_get_wtime() - timepoint));

   err = TestResults(N, checksumOrig, &checksumSorted, a);

  if(err>0)printf("%ld Errors in parsort2.\n",err);

#ifdef VERBOSE
  debug_output(N, checksumSorted, a);
#endif    
#endif
}

//
// Intern utility functions
//
void copyArr(Int *dst, Int *src, long N)
{
   for(int i=0; i<N; i++) dst[i] = src[i];
}

void clrArr(Int *arr, long N)
{
   for(int i=0; i<N; i++) arr[i] = (Int) 0;
}

void debug_output(long N, int64_t checksum, Int *a)
{
   printf("\n");
   for(int i=0;i<N;i++)printf(" %ld ",a[i]);
   printf("\n");
   printf(" checksum = %"PRId64"\n", checksum);
}

long TestResults(long N, int64_t checksumOrig, int64_t *checksumSorted, Int *a)
{
   long err     = 0;
   *checksumSorted = 0 ;
   for (int i=0; i<(N-1); i++) {
         *checksumSorted +=a[i];
         if(a[i]>a[i+1]){
             err++;
             #ifdef VERBOSE
                printf(" err in element %d values %ld  %ld \n", 
                                     i, (long) a[i], (long) a[i+1]);
             #endif
         }
   }
   *checksumSorted +=a[N-1];
   if(*checksumSorted != checksumOrig){
      printf("checksum = %"PRId64", should have been = %"PRId64"\n",
                                 *checksumSorted, checksumOrig);
   }
   return err;
}
