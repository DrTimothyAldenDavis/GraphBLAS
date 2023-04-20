//------------------------------------------------------------------------------
// GB_file.h: portable file I/O
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_FILE_H
#define GB_FILE_H

int64_t GB_file_open_and_lock   // returns position if >= 0, or -1 on error
(
    // input
    char *filename,     // full path to file to open
    // output
    FILE **fp_handle,   // file pointer of open file (NULL on error)
    int *fd_handle      // file descriptor of open file (-1 on error)
) ;

bool GB_file_unlock_and_close   // true if successful, false on error
(
    // input/output
    FILE **fp_handle,       // file pointer, set to NULL on output
    int *fd_handle          // file descriptor, set to -1 on output
) ;

bool GB_file_mkdir (char *path) ;

void *GB_file_dlopen (char *library_name) ;

void *GB_file_dlsym (void *dl_handle, char *symbol) ;

void GB_file_dlclose (void *dl_handle) ;

#endif

