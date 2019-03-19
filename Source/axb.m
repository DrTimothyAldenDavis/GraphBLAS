function axb
%AXB create all C=A*B functions for all semirings
%
% This function creates all files of the form GB_AxB__*.[ch], including 960
% semirings (GB_AxB__*.c) and one include file, GB_AxB__semirings.h.

f = fopen ('Generated/GB_AxB__semirings.h', 'w') ;
fprintf (f, '//------------------------------------------------------------------------------\n') ;
fprintf (f, '// GB_AxB__semirings.h: definitions for GB_AxB__*.c\n') ;
fprintf (f, '//------------------------------------------------------------------------------\n') ;
fprintf (f, '\n') ;
fprintf (f, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.\n') ;
fprintf (f, '// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.\n') ;
fprintf (f, '\n') ;
fprintf (f, '// This file has been automatically generated from Generator/GB_AxB.h') ;
fprintf (f, '\n\n') ;
fclose (f) ;

axb_template ('first',  1, 'xarg') ;
axb_template ('second', 1, 'yarg') ;

axb_template ('min',    0, 'GB_IMIN(xarg,yarg)', 'fminf(xarg,yarg)', 'fmin(xarg,yarg)') ;
axb_template ('max',    0, 'GB_IMAX(xarg,yarg)', 'fmaxf(xarg,yarg)', 'fmax(xarg,yarg)') ;
axb_template ('plus',   0, 'xarg + yarg') ;
axb_template ('minus',  0, 'xarg - yarg') ;
axb_template ('rminus', 0, 'yarg - xarg') ;
axb_template ('times',  0, 'xarg * yarg') ;
axb_template ('div',    0, 'GB_IDIV(xarg,yarg)', 'xarg / yarg') ;
axb_template ('rdiv',   0, 'GB_IDIV(yarg,xarg)', 'yarg / xarg') ;

axb_template ('iseq',   0, 'xarg == yarg') ;
axb_template ('isne',   0, 'xarg != yarg') ;
axb_template ('isgt',   0, 'xarg > yarg' ) ;
axb_template ('islt',   0, 'xarg < yarg' ) ;
axb_template ('isge',   0, 'xarg >= yarg') ;
axb_template ('isle',   0, 'xarg <= yarg') ;

axb_compare_template ('eq',     1, 'xarg == yarg') ;
axb_compare_template ('ne',     0, 'xarg != yarg') ;
axb_compare_template ('gt',     1, 'xarg > yarg' ) ;
axb_compare_template ('lt',     1, 'xarg < yarg' ) ;
axb_compare_template ('ge',     1, 'xarg >= yarg') ;
axb_compare_template ('le',     1, 'xarg <= yarg') ;

axb_template ('lor',    1, '(xarg != 0) || (yarg != 0)') ;
axb_template ('land',   1, '(xarg != 0) && (yarg != 0)') ;
axb_template ('lxor',   1, '(xarg != 0) != (yarg != 0)') ;

fprintf ('\n') ;
