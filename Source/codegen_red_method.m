function codegen_red_method (op, func, ztype, identity, terminal)
%CODEGEN_RED_METHOD create a reductionfunction
%
% codegen_red_method (op, func, ztype, identity, terminal)

f = fopen ('control.m4', 'w') ;

[zname, unsigned, bits] = codegen_type (ztype) ;

name = sprintf ('%s_%s', op, zname) ;

% function names
fprintf (f, 'define(`GB_red_scalar'', `GB_red_scalar__%s'')\n', name) ;

% type (no typecasting)
fprintf (f, 'define(`GB_atype'', `%s'')\n', ztype) ;
fprintf (f, 'define(`GB_ztype'', `%s'')\n', ztype) ;

% identity and terminal values for the monoid
fprintf (f, 'define(`GB_identity'', `%s'')\n', identity) ;

if (~isempty (terminal))
    fprintf (f, 'define(`GB_terminal'', `if (s == %s) break ;'')\n', ...
        terminal) ;
else
    fprintf (f, 'define(`GB_terminal'', `;'')\n') ;
end

% create the operator
func = strrep (func, 'zarg', '`$1''') ;
func = strrep (func, 'yarg', '`$2''') ;
fprintf (f, 'define(`GB_REDUCE_OP'', `%s'')\n', func) ;

fclose (f) ;

% construct the *.c file
cmd = sprintf (...
'cat control.m4 Generator/GB_red.c | m4 | tail -n +6 > Generated/GB_red__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
cmd = sprintf (...
'cat control.m4 Generator/GB_red.h | m4 | tail -n +6 >> Generated/GB_reduce__include.h') ;
system (cmd) ;

delete ('control.m4') ;

