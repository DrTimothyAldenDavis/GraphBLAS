//------------------------------------------------------------------------------
// GB_user_name_set: set the user_name of an object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_user_name_set
(
    // input/output
    char **object_user_name,        // user_name of the object
    size_t *object_user_name_size,  // user_name_size of the object
    // input
    const char *new_name            // new name for the object
)
{   GB_cov[10400]++ ;
// covered (10400): 4

    // free the object user_name, if it already exists
    GB_FREE (object_user_name, (*object_user_name_size)) ;
    (*object_user_name_size) = 0 ;

    // get the length of the new name
    size_t len = strlen (new_name) ;
    if (len == 0)
    {   GB_cov[10401]++ ;
// NOT COVERED (10401):
        // no new name; leave the object unnamed
        printf ("no new name\n") ;
        return (GrB_SUCCESS) ;
    }

    // allocate the new name
    size_t user_name_size ;
    char *user_name = GB_MALLOC (len + 1, char, &user_name_size) ;
    if (user_name == NULL)
    {   GB_cov[10402]++ ;
// NOT COVERED (10402):
        // out of memory
        printf ("out of memory\n") ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // set the new user_name
    strcpy (user_name, new_name) ;
    (*object_user_name) = user_name ;
    (*object_user_name_size) = user_name_size ;

    printf ("set name ok [%s]:%d\n", *object_user_name,
        (int) (*object_user_name_size)) ;
    return (GrB_SUCCESS) ;
}

