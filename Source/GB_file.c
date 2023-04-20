//------------------------------------------------------------------------------
// GB_file.c: portable file I/O
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods provide portable open/close/lock/unlock/mkdir functions, in
// support of the JIT.  If the JIT is disabled at compile time, these functions
// do nothing and return an error.

#include "GB.h"
#include "GB_file.h"

#ifndef NJIT

    #include <fcntl.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <errno.h>

    #if GB_COMPILER_MSC

        // MS Visual Studio
        #include <io.h>
        #include <direct.h>
        #define GB_OPEN         _open
        #define GB_CLOSE        _close
        #define GB_FDOPEN       _fdopen
        #define GB_MKDIR        _mkdir (path)
        #define GB_READ_ONLY    (_O_RDONLY)
        #define GB_WRITE_ONLY   (_O_WRONLY | _O_CREAT | _O_APPEND)
        #define GB_READ_WRITE   (_O_RDWR   | _O_CREAT | _O_APPEND)
        #define GB_MODE         (_S_IREAD | _S_IWRITE)

    #else

        // assume POSIX compliance
        #include <unistd.h>
        #define GB_OPEN         open
        #define GB_CLOSE        close
        #define GB_FDOPEN       fdopen
        #define GB_MKDIR(path)  mkdir (path, S_IRWXU)
        #define GB_READ_ONLY    (O_RDONLY)
        #define GB_WRITE_ONLY   (O_WRONLY | O_CREAT | O_APPEND)
        #define GB_READ_WRITE   (O_RDWR   | O_CREAT | O_APPEND)
        #define GB_MODE         (S_IRUSR | S_IWUSR)

    #endif

#endif

//------------------------------------------------------------------------------
// GB_file_lock:  lock a file for exclusive writing
//------------------------------------------------------------------------------

// Returns true if successful, false on error

static bool GB_file_lock (FILE *fp, int fd)
{
    #ifndef NJIT
    int result = 0 ;
    #if GB_COMPILER_MSC
    // lock file for Windows
    _lock_file (fp) ;
    #else
    // lock file for POSIX
    struct flock lock ;
    lock.l_type = F_WRLCK ;
    lock.l_whence = SEEK_SET ;
    lock.l_start = 0 ;
    lock.l_len = 0 ;
    lock.l_pid = 0 ;
    result = fcntl (fd, F_SETLKW, &lock) ;
    #endif
    return (result == 0) ;
    #else
    // JIT not enabled
    return (false) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_file_unlock:  unlock a file
//------------------------------------------------------------------------------

// Returns true if successful, false on error

static bool GB_file_unlock (FILE *fp, int fd)
{
    #ifndef NJIT
    int result = 0 ;
    #if GB_COMPILER_MSC
    // unlock file for Windows
    _unlock_file (fp) ;
    #else
    // unlock file for POSIX
    struct flock lock ;
    lock.l_type = F_UNLCK ;
    lock.l_whence = SEEK_SET ;
    lock.l_start = 0 ;
    lock.l_len = 0 ;
    lock.l_pid = 0 ;
    result = fcntl (fd, F_SETLKW, &lock) ;
    #endif
    return (result == 0) ;
    #else
    // JIT not enabled
    return (false) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_file_open_and_lock:  open and lock a file for exclusive read/write
//------------------------------------------------------------------------------

// Returns 0 if the file was newly created.  If the file already existed, this
// method seeks to the end of the file and returns the position there (a value
// > 0).  Returns -1 on error.

int64_t GB_file_open_and_lock   // returns position if >= 0, or -1 on error
(
    // input
    char *filename,     // full path to file to open
    // output
    FILE **fp_handle,   // file pointer of open file (NULL on error)
    int *fd_handle      // file descriptor of open file (-1 on error)
)
{

    #ifndef NJIT
    if (filename == NULL || fp_handle == NULL || fd_handle == NULL)
    {
        // failure: inputs invalid
        return (-1) ;
    }

    (*fp_handle) = NULL ;
    (*fd_handle) = -1 ;

    // open the file, creating it if it doesn't exist
    int fd = GB_OPEN (filename, GB_READ_WRITE, GB_MODE) ;
    if (fd == -1)
    {
        // failure: file does not exist or cannot be created
        return (-1) ;
    }

    // get the file pointer from the file descriptor
    FILE *fp = GB_FDOPEN (fd, "a") ;
    if (fp == NULL)
    {
        // failure: cannot create file pointer from file descriptor
        GB_CLOSE (fd) ;
        return (-1) ;
    }

    // lock the file
    if (!GB_file_lock (fp, fd))
    {
        // failure: cannot lock the file
        fclose (fp) ;
        return (-1) ;
    }

    // seek to the end of the file
    if (fseek (fp, 0L, SEEK_END) != 0)
    {
        // failure: cannot seek to the end of the file
        GB_file_unlock (fp, fd) ;
        fclose (fp) ;
        return (-1) ;
    }

    // get the position at the end of the file
    int64_t where = (int64_t) ftell (fp) ;
    if (where == -1)
    {
        // failure: tell the position in the file 
        GB_file_unlock (fp, fd) ;
        fclose (fp) ;
        return (-1) ;
    }

    // success: file exists, is open, and is locked for writing
    (*fp_handle) = fp ;
    (*fd_handle) = fd ;
    return (where) ;

    #else
    // JIT not enabled
    return (-1) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_file_unlock_and_close:  unlock and close a file
//------------------------------------------------------------------------------

bool GB_file_unlock_and_close   // true if successful, false on error
(
    // input/output
    FILE **fp_handle,       // file pointer, set to NULL on output
    int *fd_handle          // file descriptor, set to -1 on output
)
{

    #ifndef NJIT
    if (fp_handle == NULL || fd_handle == NULL)
    {
        // failure: inputs invalid
        return (false) ;
    }

    FILE *fp = (*fp_handle) ;
    int fd = (*fd_handle) ;

    (*fp_handle) = NULL ;
    (*fd_handle) = -1 ;

    if (fp == NULL || fd < 0)
    {
        // failure: inputs invalid
        return (false) ;
    }

    // unlock the file
    bool ok = GB_file_unlock (fp, fd) ;

    // close the file
    ok = ok && (fclose (fp) == 0) ;

    // return result
    return (ok) ;

    #else
    // JIT not enabled
    return (false) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_file_mkdir: create a directory
//------------------------------------------------------------------------------

// Create a directory, including all parent directories if they do not exist.
// Returns true if the directory already exists or if it was successfully
// created.  Returns false on error.

bool GB_file_mkdir (char *path)
{
    #ifndef NJIT
    int result = 0 ;
    if (path == NULL)
    {
        // invalid input
        return (false) ;
    }

    // create all the leading directories
    bool first = true ;
    for (char *p = path ; *p ; p++)
    {
        // look for a file separator
        if (*p == '/' || *p == '\\')
        {
            // found a file separator
            if (!first)
            { 
                // terminate the path at this file separator
                char save = *p ;
                *p = '\0' ;
                // construct the directory at this path
                result = GB_MKDIR (path) ;
                // err = (result == -1) ? errno : 0 ;
                // restore the path
                *p = save ;
            }
            first = false ;
        }
    }

    // create the final directory
    result = GB_MKDIR (path) ;
    int err = (result == -1) ? errno : 0 ;
    return (err == 0 || err == EEXIST) ;

    #else
    // JIT not enabled
    return (false) ;
    #endif
}

