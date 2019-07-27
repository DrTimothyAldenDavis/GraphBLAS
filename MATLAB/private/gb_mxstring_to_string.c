//------------------------------------------------------------------------------
// gb_mxstring_to_string: copy a MATLAB string into a C string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

int gb_mxstring_to_string   // returns length of string, or -1 if S not a string
(
    char *string,           // size maxlen
    const size_t maxlen,    // length of string
    const mxArray *S        // MATLAB mxArray containing a string
)
{

    size_t len = 0 ;
    string [0] = '\0' ;
    if (S != NULL && mxGetNumberOfElements (S) > 0)
    {
        if (!mxIsChar (S))
        {
            ERROR ("not a string") ;
        }
        len = mxGetNumberOfElements (S) ;
        if (len > 0)
        {
            mxGetString (S, string, maxlen) ;
            string [maxlen] = '\0' ;
        }
    }
    return (len) ;
}

