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
else
    fprintf (f, 'define(`GB_terminal'', `;'')\n') ;
end

% to get an entry from A
if (isequal (multop, 'second'))
    fprintf (f, 'define(`GB_geta'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_geta'', `%s aik = Ax [pA]'')\n', xytype) ;
end

% to get an entry from B
if (isequal (multop, 'first'))
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

% create the multiply-add operator
multadd = strrep (add, 't',  mult) ;
multadd = strrep (multadd, 'w', '`$1''') ;
multadd = strrep (multadd, 'xarg', '`$2''') ;
multadd = strrep (multadd, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_MULTIPLY_ADD'', `%s'')\n', multadd) ;

% create the multiply operator
mult = strrep (mult, 'xarg', '`$2''') ;
mult = strrep (mult, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_MULTIPLY'', `$1 = %s'')\n', mult) ;

% create the add operator
% (no longer used, but kept for the comments in each generated file)
add = strrep (add, 'w', '`$1''') ;
add = strrep (add, 't', '`$2''') ;
fprintf (f, 'define(`GB_ADD'', `%s'')\n', add) ;

fclose (f) ;

% construct the *.c file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.c | m4 | tail -n +15 > Generated/GB_AxB__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.h | m4 | tail -n +15 >> Generated/GB_AxB__include.h') ;
system (cmd) ;

delete ('control.m4') ;

