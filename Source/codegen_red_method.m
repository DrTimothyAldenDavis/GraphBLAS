function codegen_red_method (opname, op, atype, identity, terminal, panel)
%CODEGEN_RED_METHOD create a reduction function, C = reduce (A)
%
% codegen_red_method (opname, op, atype, identity, terminal)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[aname, unsigned, bits] = codegen_type (atype) ;

name = sprintf ('%s_%s', opname, aname) ;
is_any = isequal (opname, 'any') ;

% function names
fprintf (f, 'm4_define(`_bld'', `_bld__%s'')\n', name) ;

% the type of A, S, T, X, Y, and Z (no typecasting)
ztype = atype ;
fprintf (f, 'm4_define(`GB_atype'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_stype'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ttype'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_xtype'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ytype'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ztype'', `%s'')\n', atype) ;

is_monoid = ~isempty (identity) ;
if (is_monoid)
    % monoid function name and identity value
    fprintf (f, 'm4_define(`_red'',    `_red__%s'')\n',    name);
    fprintf (f, 'm4_define(`GB_identity'', `%s'')\n', identity) ;
else
    % first and second operators are not monoids
    fprintf (f, 'm4_define(`_red'',    `_red__(none)'')\n') ;
    fprintf (f, 'm4_define(`GB_identity'', `(none)'')\n') ;
end

% A is never iso, so GBX is not needed
fprintf (f, 'm4_define(`GB_declarea'', `%s $1'')\n', atype) ;
fprintf (f, 'm4_define(`GB_geta'', `$1 = $2 [$3]'')\n') ;

if (is_any)
    % ANY monoid is terminal
    terminal = ' ' ;
    is_terminal = 1 ;
    tvalue = '' ;
    tbreak = sprintf ('#define GB_IF_TERMINAL_BREAK(z,zterminal) break ') ;
    tcondition = sprintf ('#define GB_TERMINAL_CONDITION(z,zterminal) 1') ;
elseif (~isempty (terminal))
    % monoid is terminal
    is_terminal = 1 ;
    tbreak = sprintf ('#define GB_IF_TERMINAL_BREAK(z,zterminal) if (z == %s) { break ; }', terminal) ;
    tvalue = sprintf ('#define GB_DECLARE_MONOID_TERMINAL(modifier,zterminal) modifier %s zterminal = %s', ztype, terminal) ;
    tcondition = sprintf ('#define GB_TERMINAL_CONDITION(z,zterminal) (z == %s)', terminal) ;
else
    % monoid is not terminal
    is_terminal = 0 ;
    tvalue = '' ;
    tbreak = '' ;
    tcondition = '' ;
end

if (is_any)
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `%s'')\n', ...
        '#define GB_IS_ANY_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `'')\n') ;
end

if (is_terminal)
    fprintf (f, 'm4_define(`GB_monoid_is_terminal'', `%s'')\n', ...
        '#define GB_MONOID_IS_TERMINAL 1') ;
else
    fprintf (f, 'm4_define(`GB_monoid_is_terminal'', `'')\n') ;
end
fprintf (f, 'm4_define(`GB_terminal_condition'', `%s'')\n', tcondition) ;
fprintf (f, 'm4_define(`GB_if_terminal_break'', `%s'')\n', tbreak) ;
fprintf (f, 'm4_define(`GB_declare_monoid_terminal'', `%s'')\n', tvalue) ;

if (is_any)
    fprintf (f, 'm4_define(`GB_panel'', `(no panel)'')\n') ;
else
    fprintf (f, 'm4_define(`GB_panel'', `%d'')\n', panel) ;
end

% create the update operator
update_op = op {1} ;
update_op = strrep (update_op, 'zarg', '`$1''') ;
update_op = strrep (update_op, 'yarg', '`$2''') ;
fprintf (f, 'm4_define(`GB_update_op'', `%s'')\n', update_op) ;

% create the function operator
add_op = op {2} ;
add_op = strrep (add_op, 'zarg', '`$1''') ;
add_op = strrep (add_op, 'xarg', '`$2''') ;
add_op = strrep (add_op, 'yarg', '`$3''') ;
fprintf (f, 'm4_define(`GB_add_op'', `%s'')\n', add_op) ;

% create the disable flag
disable  = sprintf ('GxB_NO_%s', upper (opname)) ;
disable = [disable (sprintf (' || GxB_NO_%s', upper (aname)))] ;
disable = [disable (sprintf (' || GxB_NO_%s_%s', upper (opname), upper (aname)))] ;
fprintf (f, 'm4_define(`GB_disable'', `(%s)'')\n', disable) ;

fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

if (is_monoid)
    % construct the *.c file for the reduction to scalar
    cmd = sprintf ('cat control.m4 Generator/GB_red.c | m4 -P | awk -f codegen_blank.awk > Generated2/GB_red__%s.c', name) ;
    fprintf ('.') ;
    system (cmd) ;
    % append to the *.h file
    system ('cat control.m4 Generator/GB_red.h | m4 -P | awk -f codegen_blank.awk >> Generated2/GB_red__include.h') ;
end

% construct the build *.c and *.h files
cmd = sprintf ('cat control.m4 Generator/GB_bld.c | m4 -P | awk -f codegen_blank.awk > Generated2/GB_bld__%s.c', name) ;
fprintf ('.') ;
system (cmd) ;
system ('cat control.m4 Generator/GB_bld.h | m4 -P | awk -f codegen_blank.awk >> Generated2/GB_bld__include.h') ;

delete ('control.m4') ;

