//------------------------------------------------------------------------------
// GxB_Global_Option_get: get a global default option for all future matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

GrB_Info GxB_Global_Option_get      // gets the current global option
(
    GxB_Option_Field field,         // option to query
    ...                             // return value of the global option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GxB_Global_Option_get (field, &value)") ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        //----------------------------------------------------------------------
        // hyper_ratio
        //----------------------------------------------------------------------

        case GxB_HYPER : 

            va_start (ap, field) ;
            double *hyper_ratio = va_arg (ap, double *) ;
            va_end (ap) ;

            GB_RETURN_IF_NULL (hyper_ratio) ;
            (*hyper_ratio) = GB_Global_hyper_ratio_get ( ) ;
            break ;

        //----------------------------------------------------------------------
        // matrix format (CSR or CSC)
        //----------------------------------------------------------------------

        case GxB_FORMAT : 

            va_start (ap, field) ;
            GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
            va_end (ap) ;

            GB_RETURN_IF_NULL (format) ;
            (*format) = (GB_Global_is_csc_get ( )) ? GxB_BY_COL : GxB_BY_ROW ;
            break ;

        //----------------------------------------------------------------------
        // mode from GrB_init (blocking or non-blocking)
        //----------------------------------------------------------------------

        case GxB_MODE : 

            va_start (ap, field) ;
            GrB_Mode *mode = va_arg (ap, GrB_Mode *) ;
            va_end (ap) ;

            GB_RETURN_IF_NULL (mode) ;
            (*mode) = GB_Global_mode_get ( )  ;
            break ;

        //----------------------------------------------------------------------
        // threading model for synchronizing user threads
        //----------------------------------------------------------------------

        case GxB_THREAD_SAFETY : 

            va_start (ap, field) ;
            GxB_Thread_Model *thread_safety = va_arg (ap, GxB_Thread_Model *) ;
            va_end (ap) ;

            GB_RETURN_IF_NULL (thread_safety) ;
            (*thread_safety) = 

                #if defined (USER_POSIX_THREADS)
                GxB_THREAD_POSIX ;
                #elif defined (USER_WINDOWS_THREADS)
                GxB_THREAD_WINDOWS ;    // Windows threads not yet supported
                #elif defined (USER_ANSI_THREADS)
                GxB_THREAD_ANSI ;       // ANSI C11 threads not yet supported
                #elif defined ( _OPENMP ) || defined (USER_OPENMP_THREADS)
                GxB_THREAD_OPENMP ;
                #else
                GxB_THREAD_NONE ;       // GraphBLAS is not thread safe!
                #endif

            break ;

        //----------------------------------------------------------------------
        // internal parallel threading in GraphBLAS
        //----------------------------------------------------------------------

        case GxB_THREADING : 

            va_start (ap, field) ;
            GxB_Thread_Model *threading = va_arg (ap, GxB_Thread_Model *) ;
            va_end (ap) ;

            GB_RETURN_IF_NULL (threading) ;
            #if defined ( _OPENMP )
            (*threading) = GxB_THREAD_OPENMP ;
            #else
            (*threading) = GxB_THREAD_NONE ;
            #endif
            break ;

        //----------------------------------------------------------------------
        // default number of threads
        //----------------------------------------------------------------------

        case GxB_GLOBAL_NTHREADS :      // same as GxB_NTHREADS

            va_start (ap, field) ;
            int *nthreads_max = va_arg (ap, int *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (nthreads_max) ;
            (*nthreads_max) = GB_Global_nthreads_max_get ( ) ;
            break ;

        //----------------------------------------------------------------------
        // SuiteSparse:GraphBLAS version, etc
        //----------------------------------------------------------------------

        case 7 : // GxB_LIBRARY_NAME :

            va_start (ap, field) ;
            char **name = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (name) ;
            (*name) = "SuiteSparse:GraphBLAS" ;     // GxB_(whatever...) TODO 
            break ;

        case 8 : // GxB_LIBRARY_VERSION :

            va_start (ap, field) ;
            int *version = va_arg (ap, int *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (version) ;
            version [0] = GxB_IMPLEMENTATION_MAJOR ;
            version [1] = GxB_IMPLEMENTATION_MINOR ;
            version [2] = GxB_IMPLEMENTATION_SUB ;
            break ;

        case 9 : // GxB_LIBRARY_DATE :

            va_start (ap, field) ;
            char **date = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (date) ;
            (*date) = GxB_DATE ;
            break ;

        case 10 : // GxB_LIBRARY_ABOUT :

            va_start (ap, field) ;
            char **about = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (about) ;
            (*about) = GxB_ABOUT ;
            break ;

        case 11 : // GxB_LIBRARY_LICENSE :

            va_start (ap, field) ;
            char **license = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (license) ;
            (*license) = GxB_LICENSE ;
            break ;

        case 12 : // GxB_LIBRARY_COMPILE_DATE :

            va_start (ap, field) ;
            char **compile_date = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (compile_date) ;
            (*compile_date) = __DATE__ ;
            break ;

        case 13 : // GxB_LIBRARY_COMPILE_TIME :

            va_start (ap, field) ;
            char **compile_time = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (compile_time) ;
            (*compile_time) = __TIME__ ;
            break ;

        //----------------------------------------------------------------------
        // GraphBLAS API version, tec
        //----------------------------------------------------------------------

        case 14 : // GxB_API_VERSION :

            va_start (ap, field) ;
            int *api_version = va_arg (ap, int *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (api_version) ;
            api_version [0] = GxB_MAJOR ;       // TODO rename GxB_API_MAJOR ...
            api_version [1] = GxB_MINOR ;
            api_version [2] = GxB_SUB ;
            break ;

        case 15 : // GxB_API_ABOUT :

            va_start (ap, field) ;
            char **api_about = va_arg (ap, char **) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (api_about) ;
            (*api_about) = GxB_SPEC ;
            break ;

        //----------------------------------------------------------------------

/*
        case 16: // GxB_MALLOC_FUNCTION

            va_start (ap, field) ;
            void **malloc_function = va_arg (ap, void *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (malloc_function) ;
            (*malloc_function) = GB_Global_malloc_function_get ( ) ;
            break ;

        case 17: // GxB_CALLOC_FUNCTION

            va_start (ap, field) ;
            void *calloc_function = va_arg (ap, void *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (calloc_function) ;
            calloc_function = GB_Global_calloc_function_get ( ) ;
            break ;

        case 18: // GxB_REALLOC_FUNCTION

            va_start (ap, field) ;
            void *realloc_function = va_arg (ap, void *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (realloc_function) ;
            realloc_function = GB_Global_realloc_function_get ( ) ;
            break ;

        case 19: // GxB_FREE_FUNCTION

            va_start (ap, field) ;
            void *free_function = va_arg (ap, void *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (free_function) ;
            free_function = GB_Global_free_function_get ( ) ;
            break ;

        case 20: // GxB_MALLOC_IS_THREAD_SAFE

            va_start (ap, field) ;
            bool *free_function = va_arg (ap, void *) ;
            va_end (ap) ;
            GB_RETURN_IF_NULL (free_function) ;
            free_function = GB_Global_free_function_get ( ) ;
            break ;
*/

        //----------------------------------------------------------------------
        // invalid option
        //----------------------------------------------------------------------

        default : 

            return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
                    "invalid option field [%d], must be one of:\n"
                    "GxB_HYPER [%d], GxB_FORMAT [%d], GxB_MODE [%d],\n"
                    "GxB_THREAD_SAFETY [%d], GxB_THREADING [%d]"
                    "or GxB_NTHREADS [%d]",
                    (int) field, (int) GxB_HYPER, (int) GxB_FORMAT,
                    (int) GxB_MODE, (int) GxB_THREAD_SAFETY,
                    (int) GxB_THREADING, (int) GxB_NTHREADS))) ;

    }

    return (GrB_SUCCESS) ;
}

