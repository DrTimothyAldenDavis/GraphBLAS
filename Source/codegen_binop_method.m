function codegen_binop_method (binop, op, iscompare, xytype, is_binop_subset)
%CODEGEN_BINOP_METHOD create a function to compute C=binop(A,B)
%
% codegen_binop_method (binop, op, iscompare, xytype)

assert (~isequal (binop, 'any')) ;

if (nargin < 5)
    is_binop_subset = false ;
end

f = fopen ('control.m4', 'w') ;

[fname, unsigned, bits] = codegen_type (xytype) ;

name = sprintf ('%s_%s', binop, fname) ;

% function names
fprintf (f, 'define(`GB_AaddB'', `GB_AaddB__%s'')\n', name) ;
fprintf (f, 'define(`GB_AemultB'', `GB_AemultB__%s'')\n', name) ;
fprintf (f, 'define(`GB_AxD'', `GB_AxD__%s'')\n', name) ;
fprintf (f, 'define(`GB_DxB'', `GB_DxB__%s'')\n', name) ;
fprintf (f, 'define(`GB_Cdense_accumA'', `GB_Cdense_accumA__%s'')\n', name) ;
fprintf (f, 'define(`GB_Cdense_accumX'', `GB_Cdense_accumX__%s'')\n', name) ;
fprintf (f, 'define(`GB_Cdense_ewise3_noaccum'', `GB_Cdense_ewise3_noaccum__%s'')\n', name) ;

if (is_binop_subset)
    fprintf (f, 'define(`GB_Cdense_ewise3_accum'', `GB_Cdense_ewise3_accum__%s'')\n', name) ;
    fprintf (f, 'define(`if_is_binop_subset'', `'')\n') ;
    fprintf (f, 'define(`endif_is_binop_subset'', `'')\n') ;
else
    fprintf (f, 'define(`GB_Cdense_ewise3_accum'', `(none)'')\n') ;
    fprintf (f, 'define(`if_is_binop_subset'', `#if 0'')\n') ;
    fprintf (f, 'define(`endif_is_binop_subset'', `#endif'')\n') ;
end

if (isequal (binop, 'second'))
    fprintf (f, 'define(`GB_op_is_second'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_op_is_second'', `0'')\n') ;
end

% type of C, A, and B
if (iscompare)
    % GrB_LT_* and related operators are TxT->bool
    ztype = 'bool' ;
else
    % all other operators are TxT->T
    ztype = xytype ;
end
fprintf (f, 'define(`GB_ctype'', `%s'')\n', ztype) ;
fprintf (f, 'define(`GB_atype'', `%s'')\n', xytype) ;
fprintf (f, 'define(`GB_btype'', `%s'')\n', xytype) ;

% to get an entry from A
if (isequal (binop, 'second') || isequal (binop, 'pair'))
    % the value of A is ignored
    fprintf (f, 'define(`GB_geta'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_geta'', `%s $1 = $2 [$3]'')\n', xytype) ;
end

% to get an entry from B
if (isequal (binop, 'first') || isequal (binop, 'pair'))
    % the value of B is ignored
    fprintf (f, 'define(`GB_getb'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_getb'', `%s $1 = $2 [$3]'')\n', xytype) ;
end

% type-specific IDIV
if (~isempty (strfind (op, 'IDIV')))
    if (unsigned)
        op = strrep (op, 'IDIV', 'IDIV_UNSIGNED') ;
    else
        op = strrep (op, 'IDIV', 'IDIV_SIGNED') ;
    end
    op = strrep (op, ')', sprintf (', %d)', bits)) ;
end

% create the binary operator
op = strrep (op, 'xarg', '`$2''') ;
op = strrep (op, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_BINARYOP'', `$1 = %s'')\n', op) ;

% create the disable flag
disable = sprintf ('GxB_NO_%s', upper (binop)) ;
disable = [disable (sprintf (' || GxB_NO_%s', upper (fname)))] ;
disable = [disable (sprintf (' || GxB_NO_%s_%s', upper (binop), upper (fname)))] ;
fprintf (f, 'define(`GB_disable'', `(%s)'')\n', disable) ;

% ff = fopen ('temp.h', 'a') ;
% fprintf (ff, '// #define GxB_NO_%s_%s\n',  upper (binop), upper (fname)) ;
% fclose (ff) ;

fclose (f) ;

trim = 18 ;

% construct the *.c file
cmd = sprintf (...
'cat control.m4 Generator/GB_binop.c | m4 | tail -n +%d > Generated/GB_binop__%s.c', ...
trim, name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
cmd = sprintf (...
'cat control.m4 Generator/GB_binop.h | m4 | tail -n +%d >> Generated/GB_binop__include.h', trim) ;
system (cmd) ;

delete ('control.m4') ;

