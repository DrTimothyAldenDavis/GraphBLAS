function axb_method (addop, multop, add, mult, ztype, xytype, identity, terminal)
%AXB_METHOD create a function to compute C=A*B over a semiring
%
% axb_method (addop, multop, add, mult, ztype, xytype, identity, terminal)

f = fopen ('control.m4', 'w') ;

unsigned = (xytype (1) == 'u') ;

switch (xytype)
    case 'bool'
        fname = 'bool' ;
        bits = 1 ;
    case 'int8_t'
        fname = 'int8' ;
        bits = 8 ;
    case 'uint8_t'
        fname = 'uint8' ;
        bits = 8 ;
    case 'int16_t'
        fname = 'int16' ;
        bits = 16 ;
    case 'uint16_t'
        fname = 'uint16' ;
        bits = 16 ;
    case 'int32_t'
        fname = 'int32' ;
        bits = 32 ;
    case 'uint32_t'
        fname = 'uint32' ;
        bits = 32 ;
    case 'int64_t'
        fname = 'int64' ;
        bits = 64 ;
    case 'uint64_t'
        fname = 'uint64' ;
        bits = 64 ;
    case 'float'
        fname = 'fp32' ;
        bits = 32 ;
    case 'double'
        fname = 'fp64' ;
        bits = 64 ;
end

name = sprintf ('%s_%s_%s', addop, multop, fname) ;

fprintf (f, 'define(`GB_AgusB'', `GB_AgusB__%s'')\n', name) ;
fprintf (f, 'define(`GB_AdotB'', `GB_AdotB__%s'')\n', name) ;
fprintf (f, 'define(`GB_AheapB'', `GB_AheapB__%s'')\n', name) ;
fprintf (f, 'define(`GB_ztype'', `%s'')\n', ztype) ;
fprintf (f, 'define(`GB_xtype'', `%s'')\n', xytype) ;
fprintf (f, 'define(`GB_ytype'', `%s'')\n', xytype) ;
fprintf (f, 'define(`GB_identity'', `%s'')\n', identity) ;

if (~isempty (terminal))
    fprintf (f, 'define(`GB_terminal'', `if (cij == %s) break ;'')\n', ...
        terminal) ;
else
    fprintf (f, 'define(`GB_terminal'', `;'')\n') ;
end

if (isequal (multop, 'second'))
    fprintf (f, 'define(`GB_geta'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_geta'', `%s aik = Ax [pA] ;'')\n', xytype) ;
end

if (isequal (multop, 'first'))
    fprintf (f, 'define(`GB_getb'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_getb'', `%s bkj = Bx [pB] ;'')\n', xytype) ;
end

mult = strrep (mult, 'xarg', '`$2''') ;
mult = strrep (mult, 'yarg', '`$3''') ;

if (~isempty (strfind (mult, 'IDIV')))
    if (unsigned)
        mult = strrep (mult, 'IDIV', 'IDIV_UNSIGNED') ;
    else
        mult = strrep (mult, 'IDIV', 'IDIV_SIGNED') ;
    end
    mult = strrep (mult, ')', sprintf (',%d)', bits)) ;
end

fprintf (f, 'define(`GB_MULTIPLY'', `$1 = %s'')\n', mult) ;

add = strrep (add, 'w', '`$1''') ;
add = strrep (add, 't', '`$2''') ;
fprintf (f, 'define(`GB_ADD'', `%s'')\n', add) ;

fclose (f) ;

cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.c | m4 | tail -n +12 > Generated/GB_AxB__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;

cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.h | m4 | tail -n +12 >> Generated/GB_AxB__semirings.h') ;
system (cmd) ;

delete ('control.m4') ;

