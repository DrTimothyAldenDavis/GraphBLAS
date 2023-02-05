function codegen_1type_template (xtype)
%CODEGEN_1TYPE_TEMPLATE create a function to compute over a given type
%
% codegen_1type_template (xtype)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[fname, unsigned, bits] = codegen_type (xtype) ;
fprintf ('%-11s:  fname: %-7s  unsigned: %d bits: %d\n', xtype, fname, unsigned, bits) ;

% function names
fprintf (f, 'm4_define(`_Cdense_05d'', `_Cdense_05d__%s'')\n', fname) ;
fprintf (f, 'm4_define(`_Cdense_06d'', `_Cdense_06d__%s'')\n', fname) ;
fprintf (f, 'm4_define(`_Cdense_25'', `_Cdense_25__%s'')\n', fname) ;

fprintf (f, 'm4_define(`GB_ctype'', `%s'')\n', xtype) ;

% mask macro
if (isequal (xtype, 'GxB_FC32_t') || isequal (xtype, 'GxB_FC64_t'))
    asize = sprintf ('sizeof (%s)', xtype) ;
    fprintf (f, 'm4_define(`GB_ax_mask'', `GB_mcast ((GB_void *) $1, $2, %s)'')\n', asize) ;
else
    fprintf (f, 'm4_define(`GB_ax_mask'', `($1 [$2] != 0)'')\n') ;
end

% create the disable flag
disable = sprintf ('GxB_NO_%s', upper (fname)) ;
fprintf (f, 'm4_define(`GB_disable'', `(%s)'')\n', disable) ;
fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

% construct the *.c file
cmd = sprintf ('cat control.m4 Generator/GB_type.c | m4 -P > Generated2/GB_type__%s.c', fname) ;
system (cmd) ;

% append to the *.h file
system ('cat control.m4 Generator/GB_type.h | m4 -P >> Generated2/GB_type__include.h') ;

delete ('control.m4') ;

