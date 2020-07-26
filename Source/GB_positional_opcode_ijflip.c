//------------------------------------------------------------------------------
// GB_positional_opcode_ijflip: swap i and j in a positional opcode
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

// In the user view of the positional operator, i is a row index and j is a
// column index.  Inside GraphBLAS, i is an index inside a vector (i is from
// Ai [p]), and j is a vector (j = GBH (Ah, k, avlen).  If the matrix is in
// CSC format, these remain the same, and this function is not called.  If the
// matrix is in CSR format, then the positional operator must be flipped, with
// i and j trading places.

GB_Opcode GB_positional_opcode_ijflip   // return the flipped opcode
(
    GB_Opcode opcode                    // opcode to flip
)
{ 

    switch (opcode)
    {
        case GB_POSITIONI_opcode   : return (GB_POSITIONJ_opcode) ;
        case GB_POSITIONJ_opcode   : return (GB_POSITIONI_opcode) ;
        case GB_POSITIONI1_opcode  : return (GB_POSITIONJ1_opcode) ;
        case GB_POSITIONJ1_opcode  : return (GB_POSITIONI1_opcode) ;
        case GB_FIRSTI_opcode      : return (GB_FIRSTJ_opcode) ;
        case GB_FIRSTJ_opcode      : return (GB_FIRSTI_opcode) ;
        case GB_SECONDI_opcode     : return (GB_SECONDJ_opcode) ;
        case GB_SECONDJ_opcode     : return (GB_SECONDI_opcode) ;
        case GB_FIRSTI1_opcode     : return (GB_FIRSTJ1_opcode) ;
        case GB_FIRSTJ1_opcode     : return (GB_FIRSTI1_opcode) ;
        case GB_SECONDI1_opcode    : return (GB_SECONDJ1_opcode) ;
        case GB_SECONDJ1_opcode    : return (GB_SECONDI1_opcode) ;
        // non-positional opcodes are returned unmodified
        default: return (opcode) ;
    }
}

