function codegen_unop_method (unop, op, fcast, ztype, xtype)
%CODEGEN_UNOP_METHOD create a function to compute C=unop(cast(A))
%
% codegen_unop_method (unop, op, fcast, ztype, xtype)
%
%   unop: the name of the operator
%   op: a string defining the computation
%   ztype: the type of z for z=f(x)
%   xtype: the type of x for z=f(x)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[zname, zunsigned, zbits] = codegen_type (ztype) ;
[xname, xunsigned, xbits] = codegen_type (xtype) ;

name = sprintf ('%s_%s_%s', unop, zname, xname) ;

% determine if the op is identity with no typecast
is_identity = isequal (unop, 'identity') ;
no_typecast = isequal (ztype, xtype) ;
if (is_identity && no_typecast)
    % disable the _unop_apply method
    fprintf (f, 'm4_define(`_unop_apply'', `_unop_apply__(none)'')\n') ;
    fprintf (f, 'm4_define(`if_unop_apply_enabled'', `-1'')\n') ;
else
    fprintf (f, 'm4_define(`_unop_apply'', `_unop_apply__%s'')\n', name) ;
    fprintf (f, 'm4_define(`if_unop_apply_enabled'', `0'')\n') ;
end

% function names
fprintf (f, 'm4_define(`_unop_tran'', `_unop_tran__%s'')\n', name) ;

% type of C and A
fprintf (f, 'm4_define(`GB_ctype'', `%s'')\n', ztype) ;
fprintf (f, 'm4_define(`GB_atype'', `%s'')\n', xtype) ;

A_is_pattern = (isempty (strfind (op, 'xarg'))) ;

% to get an entry from A
if (A_is_pattern)
    % A(i,j) is not needed
    fprintf (f, 'm4_define(`GB_declarea'', `;'')\n') ;
    fprintf (f, 'm4_define(`GB_geta'', `;'')\n') ;
else
    % A is not iso, so GBX (Ax, p, A->iso) is not needed
    fprintf (f, 'm4_define(`GB_declarea'', `%s $1'')\n', xtype) ;
    fprintf (f, 'm4_define(`GB_geta'', `$1 = $2 [$3]'')\n') ;
end

% type-specific iminv
if (~isempty (strfind (op, 'iminv')))
    if (zunsigned)
        op = strrep (op, 'iminv (', sprintf ('idiv_uint%d (1, ', zbits)) ;
    else
        op = strrep (op, 'iminv (', sprintf ('idiv_int%d (1, ', zbits)) ;
    end
end

% create the unary operator
op = strrep (op, 'xarg', '`$2''') ;
fprintf (f, 'm4_define(`GB_unaryop'', `$1 = %s'')\n', op) ;

% create the cast operator
if (A_is_pattern)
    % cast (A(i,j)) is not needed
    fprintf (f, 'm4_define(`GB_cast'', `;'')\n') ;
else
    fcast = strrep (fcast, 'zarg', '`$1''') ;
    fcast = strrep (fcast, 'xarg', '`$2''') ;
    fprintf (f, 'm4_define(`GB_cast'', `%s'')\n', fcast) ;
end

% create the disable flag
disable  = sprintf ('GxB_NO_%s', upper (unop)) ;
disable = [disable (sprintf (' || GxB_NO_%s', upper (zname)))] ;
if (~isequal (zname, xname))
    disable = [disable (sprintf (' || GxB_NO_%s', upper (xname)))] ;
end
fprintf (f, 'm4_define(`GB_disable'', `(%s)'')\n', disable) ;
fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

% construct the *.c file
cmd = sprintf ('cat control.m4 Generator/GB_unop.c | m4 -P | awk -f codegen_blank.awk > Generated2/GB_unop__%s.c', name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
system ('cat control.m4 Generator/GB_unop.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> Generated2/GB_unop__include.h') ;

delete ('control.m4') ;

