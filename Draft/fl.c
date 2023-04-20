#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>

int main (void)
{
    pid_t me = getpid ( ) ;
    printf ("I am %d\n", me) ;

    //--------------------------------------------------------------------------
    // open the source file, creating it if it doesn't exist
    //--------------------------------------------------------------------------

    int fd = open ("stuff.c", O_RDWR /* O_WRONLY */ | O_CREAT | O_APPEND,
        S_IRUSR | S_IWUSR) ;
    perror ("open:") ;
    printf ("fd : %d %d %d\n", fd, errno, EEXIST) ;

    //--------------------------------------------------------------------------
    // set a write lock on the source file
    //--------------------------------------------------------------------------

    struct flock lock ;
    lock.l_type = F_WRLCK ;
    lock.l_whence = SEEK_SET ;
    lock.l_start = 0 ;
    lock.l_len = 0 ;
    lock.l_pid = 0 ;

    printf ("set the write lock ...\n") ;
    int result = fcntl (fd, F_SETLKW, &lock) ;
    perror ("fcntl LOCK:") ;
    printf ("result %d, sleeping ...\n", result) ;

    FILE *fp = fdopen (fd, "a") ;
    result = fseek (fp, 0L, SEEK_END) ;
    perror ("seek:") ;
    long where = (size_t) ftell (fp) ;
    perror ("where:") ;
    if (where == 0)
    {

        //----------------------------------------------------------------------
        // write the source file and compile it
        //--------------------------------------------------------------------- 

        printf ("I own it: %ld\n", where) ;
        fprintf (fp, "hello this is %d\n", me) ;
        fflush (fp) ;
        perror ("flush:") ;

    }
    else
    {
        printf ("I do not own it: %ld\n", where) ;
        printf ("darn, someone wrote it already\n") ;

    }
    sleep (20) ;

    //--------------------------------------------------------------------------
    // read the library.so file
    //--------------------------------------------------------------------------

    // dlopen goes here

    //--------------------------------------------------------------------------
    // release the lock on the source file
    //--------------------------------------------------------------------------

    printf ("unlock the write lock ...\n") ;
    lock.l_type = F_UNLCK ;
    result = fcntl (fd, F_SETLKW, &lock) ;
    perror ("unlock:") ;

    //--------------------------------------------------------------------------
    // close the file
    //--------------------------------------------------------------------------

    printf ("result %d, bye ...\n", result) ;
    fclose (fp) ;
    perror ("flush:") ;

}

