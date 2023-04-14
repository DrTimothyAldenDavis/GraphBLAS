
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

typedef struct { double xreal ; double ximag ; } GxB_FC64_struct_t ;

#define GB_PUN(type,value) (*((type *) (&(value))))

#if defined ( __cplusplus )
#define GB_cabs(x) std::abs (GB_PUN (std::complex<double>, x))
#define GB_clog(x) GB_to_native (std::log (GB_PUN (std::complex<double>, x)))
#define GB_native std::complex<double>
#include <cmath>
#include <complex>
#else
#include <complex.h>
#define GB_cabs(x) cabs (GB_PUN (double complex, x))
#define GB_clog(x) GB_to_native (clog (GB_PUN (double complex, x)))
#define GB_native double complex
#endif

    static inline GxB_FC64_struct_t GB_complex (double xreal, double ximag)
    {
        GxB_FC64_struct_t z ;
        z.xreal = xreal ;
        z.ximag = ximag ;
        return (z) ;
    }

    static inline GxB_FC64_struct_t GB_to_native (GB_native x)
    {
        GxB_FC64_struct_t z = GB_PUN (GxB_FC64_struct_t, x) ;
        return (z) ;
    }

#define GB_CMPLX64(x,y) GB_complex (x,y)

int main (void)
{
    #if defined ( __cplusplus )
    printf ("hello, this is C++\n") ;
    #else
    printf ("hello, this is C\n") ;
    #endif

    GxB_FC64_struct_t x = { .xreal = 3.4, .ximag = -0.9 } ;

    GB_native y ;

    double a = GB_cabs (x) ;
    double b = sqrt ((3.4)*(3.4) + (.9)*(.9)) ;
    printf ("a %g err: %g\n", a, a-b) ;

    GxB_FC64_struct_t w = GB_CMPLX64 (a,b+2) ;

    y = GB_PUN (GB_native, w) ;

    GxB_FC64_struct_t z = GB_PUN (GxB_FC64_struct_t, y) ;
    printf ("z (%32.16g,%32.16g)\n", z.xreal, z.ximag) ;

    GxB_FC64_struct_t t = GB_clog (z) ;
    printf ("t (%32.16g,%32.16g)\n", t.xreal, t.ximag) ;


}

