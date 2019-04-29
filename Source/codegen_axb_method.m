function codegen_axb_method (addop, multop, add, mult, ztype, xytype, identity, terminal)
%CODEGEN_AXB_METHOD create a function to compute C=A*B over a semiring
%
% codegen_axb_method (addop, multop, add, mult, ztype, xytype, identity, terminal)

f = fopen ('control.m4', 'w') ;

[fname, unsigned, bits] = codegen_type (xytype) ;

name = sprintf ('%s_%s_%s', addop, multop, fname) ;

% function names
fprintf (f, 'define(`GB_AgusB'', `GB_AgusB__%s'')\n', name) ;
fprintf (f, 'define(`GB_AdotB'', `GB_AdotB__%s'')\n', name) ;
fprintf (f, 'define(`GB_Adot2B'', `GB_Adot2B__%s'')\n', name) ;
fprintf (f, 'define(`GB_AheapB'', `GB_AheapB__%s'')\n', name) ;

% type of C, A, and B
fprintf (f, 'define(`GB_ctype'', `%s'')\n', ztype) ;
fprintf (f, 'define(`GB_atype'', `%s'')\n', xytype) ;
fprintf (f, 'define(`GB_btype'', `%s'')\n', xytype) ;

% identity and terminal values for the monoid
fprintf (f, 'define(`GB_identity'', `%s'')\n', identity) ;

if (~isempty (terminal))
    fprintf (f, 'define(`GB_terminal'', `if (cij == %s) break ;'')\n', ...
        terminal) ;
    fprintf (f, 'define(`GB_dot_simd'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_terminal'', `;'')\n') ;
    fprintf (f, 'define(`GB_dot_simd'', `GB_PRAGMA_SIMD'')\n') ;
end

% to get an entry from A
is_second = isequal (multop, 'second') ;
if (is_second)
    fprintf (f, 'define(`GB_geta'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_geta'', `%s aik = Ax [pA]'')\n', xytype) ;
end

% to get an entry from B
is_first = isequal (multop, 'first') ;
if (is_first)
    fprintf (f, 'define(`GB_getb'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_getb'', `%s bkj = Bx [pB]'')\n', xytype) ;
end

% type-specific IDIV
if (~isempty (strfind (mult, 'IDIV')))
    if (unsigned)
        mult = strrep (mult, 'IDIV', 'IDIV_UNSIGNED') ;
    else
        mult = strrep (mult, 'IDIV', 'IDIV_SIGNED') ;
    end
    mult = strrep (mult, ')', sprintf (', %d)', bits)) ;
end

% create the multiply operator
mult2 = strrep (mult,  'xarg', '`$2''') ;
mult2 = strrep (mult2, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_MULTIPLY'', `$1 = %s'')\n', mult2) ;

% create the add operator
add2 = strrep (add,  'w', '`$1''') ;
add2 = strrep (add2, 't', '`$2''') ;
fprintf (f, 'define(`GB_ADD'', `%s'')\n', add2) ;

% create the multiply-add operator
if (isequal (ztype, 'float') || isequal (ztype, 'double') || ...
    isequal (ztype, 'bool') || is_first || is_second || ...
    isequal (multop (1:2), 'is'))
    % float and double do not get promoted.
    % bool is OK since promotion of the result (0 or 1) to int is safe.
    % first and second are OK since no promotion occurs.
    % is* operators are OK too.
    multadd = strrep (add, 't',  mult) ;
    multadd = strrep (multadd, 'w', '`$1''') ;
    multadd = strrep (multadd, 'xarg', '`$2''') ;
    multadd = strrep (multadd, 'yarg', '`$3''') ;
    fprintf (f, 'define(`GB_MULTIPLY_ADD'', `%s'')\n', multadd) ;
else
    % use explicit typecasing to avoid ANSI C integer promotion.
    add2 = strrep (add,  'w', '`$1''') ;
    add2 = strrep (add2, 't', 'x_op_y') ;
    fprintf (f, 'define(`GB_ADD'', `%s'')\n', add2) ;
    fprintf (f, 'define(`GB_MULTIPLY_ADD'', `%s x_op_y = %s ; %s'')\n', ...
        ztype, mult2, add2) ;
end

fclose (f) ;

% construct the *.c file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.c | m4 | tail -n +16 > Generated/GB_AxB__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.h | m4 | tail -n +16 >> Generated/GB_AxB__include.h') ;
system (cmd) ;

delete ('control.m4') ;

