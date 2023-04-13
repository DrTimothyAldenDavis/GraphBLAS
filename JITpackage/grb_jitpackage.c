//------------------------------------------------------------------------------
// grb_jitpackage: package GraphBLAS source code for the JIT 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// zstd.h include file
//------------------------------------------------------------------------------

// ZSTD uses switch statements with no default case.
#pragma GCC diagnostic ignored "-Wswitch-default"

// disable ZSTD deprecation warnings and include all ZSTD definitions  

// GraphBLAS does not use deprecated functions, but the warnings pop up anyway
// when GraphBLAS is built, so silence them with this #define:
#define ZSTD_DISABLE_DEPRECATE_WARNINGS

// do not use multithreading in ZSTD itself.
#undef ZSTD_MULTITHREAD

// do not use asm
#define ZSTD_DISABLE_ASM

#include "zstd.h"

//------------------------------------------------------------------------------
// zstd source (subset used by GraphBLAS)
//------------------------------------------------------------------------------

// Include the unmodified zstd, version 1.5.3.  This ensures that any files
// compressed with grb_jitpackage can be uncompressed with the same zstd
// subset that resides within libgraphblas.so itself.

#include "zstd_subset/common/debug.c"
#include "zstd_subset/common/entropy_common.c"
#include "zstd_subset/common/error_private.c"
#include "zstd_subset/common/fse_decompress.c"
#include "zstd_subset/common/pool.c"
#include "zstd_subset/common/threading.c"
#include "zstd_subset/common/xxhash.c"
#include "zstd_subset/common/zstd_common.c"

#include "zstd_subset/compress/fse_compress.c"
#include "zstd_subset/compress/hist.c"
#include "zstd_subset/compress/huf_compress.c"
#include "zstd_subset/compress/zstd_compress.c"
#include "zstd_subset/compress/zstd_compress_literals.c"
#include "zstd_subset/compress/zstd_compress_sequences.c"
#include "zstd_subset/compress/zstd_compress_superblock.c"
#include "zstd_subset/compress/zstd_double_fast.c"
#include "zstd_subset/compress/zstd_fast.c"
#include "zstd_subset/compress/zstd_lazy.c"
#include "zstd_subset/compress/zstd_ldm.c"
#include "zstd_subset/compress/zstdmt_compress.c"
#include "zstd_subset/compress/zstd_opt.c"

/* no need for decompression here
#include "zstd_subset/decompress/huf_decompress.c"
#include "zstd_subset/decompress/zstd_ddict.c"
#include "zstd_subset/decompress/zstd_decompress_block.c"
#include "zstd_subset/decompress/zstd_decompress.c"
*/

//------------------------------------------------------------------------------
// grb_prepackage main program
//------------------------------------------------------------------------------

#define OK(x) if (!(x)) { printf ("Error line %d\n", __LINE__) ; abort ( ) ; }

int main (int argc, char **argv)
{
    for (int k = 1 ; k < argc ; k++)
    {
        // read the input file
        FILE *fp = fopen (argv [k], "r") ;
        OK (fp != NULL) ;

        // find the output file name
        char *name = argv [k] ;
        for (char *p = argv [k] ; *p != '\0' ; p++)
        {
            if (*p == '/')
            {
                name = p + 1 ;
            }
        }

        // find the length of the file
        fseek (fp, 0, SEEK_END) ;
        size_t inputsize = ftell (fp) ;
        rewind (fp) ;

        OK (inputsize > 0) ;

        // read the file
        char *input = malloc (inputsize+2) ;
        OK (input != NULL) ;
        size_t nread = fread (input, sizeof (char), inputsize, fp) ;
        input [inputsize] = '\0' ; 
        fclose (fp) ;

        // allocate the compressed dst
        size_t dbound = ZSTD_compressBound (inputsize) ;
        uint8_t *dst = malloc (dbound+2) ;
        OK (dst != NULL) ;

        // compress the input file into dst using the highest compression level
        size_t dsize = ZSTD_compress (dst, dbound+2, input, inputsize, 19) ;

        // create the output file 
        char filename [1024] ;
        size_t newlen = strlen (name) ;
        char *newname = malloc (newlen+1) ;

        for (int kk = 0 ; kk < newlen ; kk++)
        {
            newname [kk] = (name [kk] == '.') ? '_' : name [kk] ;
        }
        newname [newlen] = '\0' ;

        snprintf (filename, 1024, "GB_JITpackage_%s.c", newname) ;
        fp = fopen (filename, "w") ;

        fprintf (fp, "#include \"GB_JITpackage.h\"\n") ;
        fprintf (fp, "const char   *GB_JITpackage_%s_name = \"%s\" ;\n",
            newname, name) ;
        fprintf (fp, "const uint8_t GB_JITpackage_%s [%lu] = {\n",
            newname, dsize) ;
        printf ("const size_t   GB_JITpackage_%s_len = %lu ;\n",
            newname, dsize) ;
        printf ("const char    *GB_JITpackage_%s ;\n", newname) ;
        printf ("const uint8_t *GB_JITpackage_%s ;\n", newname) ;

        for (int64_t k = 0 ; k < dsize ; k++)
        {
            fprintf (fp, "%3d,", dst [k]) ;
            if (k % 20 == 19) fprintf (fp, "\n") ;
        }
        fprintf (fp, "\n} ;\n") ;
        fclose (fp) ;

        free (newname) ;
        free (dst) ;
        free (input) ;
    }
}

