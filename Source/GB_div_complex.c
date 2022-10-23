//------------------------------------------------------------------------------
// complex division
//------------------------------------------------------------------------------

// z = x/y where z, x, and y are double complex.  The real and imaginary parts
// are passed as separate arguments to this routine.  The NaN case is ignored
// for the double relop yr >= yi.  Returns 1 if the denominator is zero, 0
// otherwise.
//
// This uses ACM Algo 116, by R. L. Smith, 1962, which tries to avoid underflow
// and overflow.
//
// z can be aliased with x or y.
//
// Note that this function has the same signature as SuiteSparse_divcomplex.

#include "GB.h"

int GB_divcomplex
(
    double xr, double xi,       // real and imaginary parts of x
    double yr, double yi,       // real and imaginary parts of y
    double *zr, double *zi      // real and imaginary parts of z
)
{

    double tr, ti, r, den ;
    int yr_class = fpclassify (yr) ;
    int yi_class = fpclassify (yi) ;
    if (yi_class == FP_ZERO)
    {

        // (zr,zi) = (xr,xi) / (yr,0)
        den = yr ;
        if (xi == 0)
        {
            tr = xr / den ;
            ti = 0 ;
        }
        else if (xr == 0)
        {
            tr = 0 ;
            ti = xi / den ;
        }
        else
        {
            tr = xr / den ;
            ti = xi / den ;
        }

    }
    else if (yr_class == FP_ZERO)
    {

        // (zr,zi) = (xr,xi) / (0,yi)
        //         = (1i * (xr, xi)) / (1i * (0,yi)
        //         =      (-xi, xr)) / (-yi,0)
        //         =       (xi,-xr)  / (yi,0)
        den = yi ;
        if (xr == 0)
        {
            tr = xi / den ;
            ti = 0 ;
        }
        else if (xi == 0)
        {
            tr = 0 ;
            ti = -xr / den ;
        }
        else
        {
            tr = xi / den ;
            ti = -xr / den ;
        }

    }
    else

#if 0
    if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)
    {

        // Smith
        r = (signbit (yr) == signbit (yi)) ? (1) : (-1) ;
        den = yr + r * yi ;
        tr = (xr + xi * r) / den ;
        ti = (xi - xr * r) / den ;

    }
    else
#endif

    {

        GxB_FC64_t x = GxB_CMPLX (xr, xi) ;
        GxB_FC64_t y = GxB_CMPLX (yr, yi) ;
        GxB_FC64_t z = x / y ;
        tr = creal (z) ;
        ti = cimag (z) ;

#if 0
        // Smith
        if (fabs (yr) >= fabs (yi))
        {
            r = yi / yr ;
            den = yr + r * yi ;
            tr = (xr + xi * r) / den ;
            ti = (xi - xr * r) / den ;
        }
        else
        {
            r = yr / yi ;
            den = r * yr + yi ;
            tr = (xr * r + xi) / den ;
            ti = (xi * r - xr) / den ;
        }
#endif

    }

    (*zr) = tr ;
    (*zi) = ti ;
    return (den == 0) ;
}

