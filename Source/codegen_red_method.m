function codegen_red_method (opname, op, atype, identity, terminal, panel)
%CODEGEN_RED_METHOD create a reduction function, C = reduce (A)
%
% codegen_red_method (opname, op, atype, identity, terminal)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;

[aname, unsigned, bits] = codegen_type (atype) ;

name = sprintf ('%s_%s', opname, aname) ;
is_any = isequal (opname, 'any') ;

% function names
fprintf (f, 'define(`_bld'', `_bld__%s'')\n', name) ;

% the type of A, S, T, X, Y, and Z (no typecasting)
fprintf (f, 'define(`GB_atype'', `%s'')\n', atype) ;
fprintf (f, 'define(`GB_stype'', `%s'')\n', atype) ;
fprintf (f, 'define(`GB_ttype'', `%s'')\n', atype) ;
fprintf (f, 'define(`GB_xtype'', `%s'')\n', atype) ;
fprintf (f, 'define(`GB_ytype'', `%s'')\n', atype) ;
fprintf (f, 'define(`GB_ztype'', `%s'')\n', atype) ;

is_monoid = ~isempty (identity) ;
if (is_monoid)
    % monoid function name and identity value
    fprintf (f, 'define(`_red'',    `_red__%s'')\n',    name);
    fprintf (f, 'define(`GB_identity'', `%s'')\n', identity) ;
else
    % first and second operators are not monoids
    fprintf (f, 'define(`_red'',    `_red__(none)'')\n') ;
    fprintf (f, 'define(`GB_identity'', `(none)'')\n') ;
end

% A is never iso, so GBX is not needed
fprintf (f, 'define(`GB_declarea'', `%s $1'')\n', atype) ;
fprintf (f, 'define(`GB_geta'', `$1 = $2 [$3]'')\n') ;

if (is_any)
    fprintf (f, 'define(`GB_is_any_monoid'', `1'')\n') ;
    fprintf (f, 'define(`GB_monoid_is_terminal'', `1'')\n') ;
    fprintf (f, 'define(`GB_terminal_value'', `(any value)'')\n') ;
    fprintf (f, 'define(`GB_terminal_condition'', `true'')\n') ;
    fprintf (f, 'define(`GB_if_terminal_break'', `break ;'')\n') ;
elseif (~isempty (terminal))
    fprintf (f, 'define(`GB_is_any_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_monoid_is_terminal'', `1'')\n') ;
    fprintf (f, 'define(`GB_terminal_value'', `%s'')\n', terminal) ;
    fprintf (f, 'define(`GB_terminal_condition'', `(`$1'' == %s)'')\n', terminal) ;
    fprintf (f, 'define(`GB_if_terminal_break'', `if (`$1'' == %s) { break ; }'')\n', terminal) ;
else
    fprintf (f, 'define(`GB_is_any_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_monoid_is_terminal'', `0'')\n') ;
    fprintf (f, 'define(`GB_terminal_value'', `(none)'')\n') ;
    fprintf (f, 'define(`GB_terminal_condition'', `(false)'')\n') ;
    fprintf (f, 'define(`GB_if_terminal_break'', `;'')\n') ;
end

if (is_any)
    fprintf (f, 'define(`GB_panel'', `(no panel)'')\n') ;
else
    fprintf (f, 'define(`GB_panel'', `%d'')\n', panel) ;
end

% create the update operator
update_op = op {1} ;
update_op = strrep (update_op, 'zarg', '`$1''') ;
update_op = strrep (update_op, 'yarg', '`$2''') ;
fprintf (f, 'define(`GB_update_op'', `%s'')\n', update_op) ;

% create the function operator
add_op = op {2} ;
add_op = strrep (add_op, 'zarg', '`$1''') ;
add_op = strrep (add_op, 'xarg', '`$2''') ;
add_op = strrep (add_op, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_add_op'', `%s'')\n', add_op) ;

% create the disable flag
disable  = sprintf ('GxB_NO_%s', upper (opname)) ;
disable = [disable (sprintf (' || GxB_NO_%s', upper (aname)))] ;
disable = [disable (sprintf (' || GxB_NO_%s_%s', upper (opname), upper (aname)))] ;
fprintf (f, 'define(`GB_disable'', `(%s)'')\n', disable) ;

fclose (f) ;

if (is_monoid)
    % construct the *.c file for the reduction to scalar
    cmd = sprintf (...
    'cat control.m4 Generator/GB_red.c | m4 | tail -n +21 > Generated2/GB_red__%s.c', ...
    name) ;
    fprintf ('.') ;
    system (cmd) ;
    % append to the *.h file
    cmd = sprintf (...
    'cat control.m4 Generator/GB_red.h | m4 | tail -n +21 >> Generated2/GB_red__include.h') ;
    system (cmd) ;
end

% construct the build *.c and *.h files
cmd = sprintf (...
'cat control.m4 Generator/GB_bld.c | m4 | tail -n +21 > Generated2/GB_bld__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;
cmd = sprintf (...
'cat control.m4 Generator/GB_bld.h | m4 | tail -n +21 >> Generated2/GB_bld__include.h') ;
system (cmd) ;

delete ('control.m4') ;

