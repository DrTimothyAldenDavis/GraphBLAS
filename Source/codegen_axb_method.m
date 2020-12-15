function codegen_axb_method (addop, multop, add, addfunc, mult, ztype, ...
    xytype, identity, terminal, omp_atomic, omp_microsoft_atomic)
%CODEGEN_AXB_METHOD create a function to compute C=A*B over a semiring
%
% codegen_axb_method (addop, multop, add, addfunc, mult, ztype, xytype, identity, terminal, omp_atomic, omp_microsoft_atomic)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isempty (mult))
    return
end

f = fopen ('control.m4', 'w') ;

is_first  = false ;
is_second = false ;
is_pair   = false ;
is_positional = false ;
switch (multop)
    case { 'firsti', 'firsti1', 'firstj', 'firstj1', 'secondj', 'secondj1' }
        is_positional = true ;
    case { 'first' }
        is_first = true ;
    case { 'second' }
        is_second = true ;
    case { 'pair' }
        is_pair = true ;
end

is_any = isequal (addop, 'any') ;
is_max = isequal (addop, 'max') ;
is_min = isequal (addop, 'min') ;
is_eq  = isequal (addop, 'eq') ;
is_any_pair   = is_any && isequal (multop, 'pair') ;
ztype_is_real = ~contains (ztype, 'FC') ;
is_plus_pair_real = isequal (addop, 'plus') && isequal (multop, 'pair') ...
    && ztype_is_real ;

switch (ztype)
    case { 'bool' }
        ztype_ignore_overflow = false ;
        nbits = 8 ;
        bits = '0x1L' ;
    case { 'int8_t', 'uint8_t' }
        ztype_ignore_overflow = false ;
        nbits = 8 ;
        bits = '0xffL' ;
    case { 'int16_t', 'uint16_t' }
        ztype_ignore_overflow = false ;
        nbits = 16 ;
        bits = '0xffffL' ;
    case { 'int32_t', 'uint32_t' }
        ztype_ignore_overflow = false ;
        nbits = 32 ;
        bits = '0xffffffffL' ;
    case { 'int64_t', 'uint64_t' }
        ztype_ignore_overflow = true ;
        nbits = 64 ;
        bits = '0' ;
    case { 'float' }
        ztype_ignore_overflow = true ;
        nbits = 32 ;
        bits = '0' ;
    case { 'double', 'GxB_FC32_t' }
        ztype_ignore_overflow = true ;
        nbits = 64 ;
        bits = '0' ;
    case { 'GxB_FC64_t' }
        ztype_ignore_overflow = true ;
        nbits = 128 ;
        bits = '0' ;
    otherwise
        error ('unknown type') ;
end

% bits: special cases for the PAIR multiplier
fprintf (f, 'define(`GB_ctype_bits'', `%s'')\n', bits) ;

% nbits: # of bits in the type, needed for the atomic compare-exchange:
fprintf (f, 'define(`GB_atomic_compare_exchange'', `GB_ATOMIC_COMPARE_EXCHANGE_%d'')\n', nbits) ;

if (is_pair)
    % these semirings are renamed to any_pair, and not thus created
    if (isequal (addop, 'land') || isequal (addop, 'eq'   ) || ...
        isequal (addop, 'lor' ) || isequal (addop, 'max'  ) || ...
        isequal (addop, 'min' ) || isequal (addop, 'times'))
        return
    end
end

[fname, unsigned, bits] = codegen_type (xytype) ;
[zname, ~, ~] = codegen_type (ztype) ;

name = sprintf ('%s_%s_%s', addop, multop, fname) ;

% function names
fprintf (f, 'define(`GB_Adot2B'', `GB_Adot2B__%s'')\n', name) ;
fprintf (f, 'define(`GB_Adot3B'', `GB_Adot3B__%s'')\n', name) ;
fprintf (f, 'define(`GB_Adot4B'', `GB_Adot4B__%s'')\n', name) ;
fprintf (f, 'define(`GB_Asaxpy3B'', `GB_Asaxpy3B__%s'')\n', name) ;

% type of C, A, and B
fprintf (f, 'define(`GB_ctype'', `%s'')\n', ztype) ;
fprintf (f, 'define(`GB_atype'', `%s'')\n', xytype) ;
fprintf (f, 'define(`GB_btype'', `%s'')\n', xytype) ;

% flag if ztype can ignore overflow in some computations
fprintf (f, 'define(`GB_ctype_ignore_overflow'', `%d'')\n', ztype_ignore_overflow) ;

% simple typecast from 1 (or 2) real scalars to any other type
switch (ztype)
    case { 'GxB_FC32_t' }
        fprintf (f, 'define(`GB_ctype_cast'', `GxB_CMPLXF (((float) $1), ((float) $2))'')\n') ;
    case { 'GxB_FC64_t' }
        fprintf (f, 'define(`GB_ctype_cast'', `GxB_CMPLX (((double) $1), ((double) $2))'')\n') ;
    otherwise
        fprintf (f, 'define(`GB_ctype_cast'', `((GB_ctype) $1)'')\n') ;
end

% identity and terminal values for the monoid
fprintf (f, 'define(`GB_identity'', `%s'')\n', identity) ;

if (is_any_pair)
    fprintf (f, 'define(`GB_is_any_pair_semiring'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_any_pair_semiring'', `0'')\n') ;
end

if (is_plus_pair_real)
    fprintf (f, 'define(`GB_is_plus_pair_real_semiring'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_plus_pair_real_semiring'', `0'')\n') ;
end

if (is_pair)
    fprintf (f, 'define(`GB_is_pair_multiplier'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_pair_multiplier'', `0'')\n') ;
end

if (is_eq)
    fprintf (f, 'define(`GB_is_eq_monoid'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_eq_monoid'', `0'')\n') ;
end

if (is_any)
    % the ANY monoid terminates on the first entry seen
    fprintf (f, 'define(`GB_is_any_monoid'', `1'')\n') ;
    fprintf (f, 'define(`GB_terminal'', `break ;'')\n') ;
    fprintf (f, 'define(`GB_dot_simd_vectorize'', `;'')\n') ;
elseif (~isempty (terminal))
    % terminal monoids terminate when cij equals the terminal value
    fprintf (f, 'define(`GB_is_any_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_terminal'', `if (cij == %s) break ;'')\n', ...
        terminal) ;
    fprintf (f, 'define(`GB_dot_simd_vectorize'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_is_any_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_terminal'', `;'')\n') ;
    fprintf (f, 'define(`GB_terminal_flag'', `;'')\n') ;
    op = '' ;
    if (ztype_is_real)
        switch (addop)
            case { 'plus' }
                op = '+' ;
            case { 'times' }
                op = '*' ;
            case { 'lor' }
                op = '||' ;
            case { 'land' }
                op = '&&' ;
            case { 'lxor' }
                op = '^' ;
            case { 'bor' }
                op = '|' ;
            case { 'band' }
                op = '&' ;
            case { 'bxor' }
                op = '^' ;
            otherwise
                op = '' ;
        end
    end
    if (isempty (op))
        fprintf (f, 'define(`GB_dot_simd_vectorize'', `;'')\n') ;
    else
        pragma = sprintf ('GB_PRAGMA_SIMD_REDUCTION (%s,$1)', op) ;
        fprintf (f, 'define(`GB_dot_simd_vectorize'', `%s'')\n', pragma) ;
    end
end

if (ztype_is_real)
    % all built-in real monoids are atomic
    fprintf (f, 'define(`GB_has_atomic'', `1'')\n') ;
else
    % complex monoids are not atomic, except for 'plus'
    if (isequal (addop, 'plus'))
        fprintf (f, 'define(`GB_has_atomic'', `1'')\n') ;
    else
        fprintf (f, 'define(`GB_has_atomic'', `0'')\n') ;
    end
end

% firsti multiply operator
if (contains (multop, 'firsti'))
    fprintf (f, 'define(`GB_is_firsti_multiplier'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_firsti_multiplier'', `0'')\n') ;
end

% firstj multiply operator
if (contains (multop, 'firstj'))
    fprintf (f, 'define(`GB_is_firstj_multiplier'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_firstj_multiplier'', `0'')\n') ;
end

% secondj multiply operator
if (contains (multop, 'secondj'))
    fprintf (f, 'define(`GB_is_secondj_multiplier'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_secondj_multiplier'', `0'')\n') ;
end

% plus_fc32 monoid:
if (isequal (addop, 'plus') && isequal (ztype, 'GxB_FC32_t'))
    fprintf (f, 'define(`GB_is_plus_fc32_monoid'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_plus_fc32_monoid'', `0'')\n') ;
end

% plus_fc64 monoid:
if (isequal (addop, 'plus') && isequal (ztype, 'GxB_FC64_t'))
    fprintf (f, 'define(`GB_is_plus_fc64_monoid'', `1'')\n') ;
else
    fprintf (f, 'define(`GB_is_plus_fc64_monoid'', `0'')\n') ;
end

% min monoids:
if (is_min)
    if (contains (ztype, 'int'))
        % min monoid for signed or unsigned integers
        fprintf (f, 'define(`GB_is_imin_monoid'', `1'')\n') ;
        fprintf (f, 'define(`GB_is_fmin_monoid'', `0'')\n') ;
    else
        % min monoid for float or double
        fprintf (f, 'define(`GB_is_imin_monoid'', `0'')\n') ;
        fprintf (f, 'define(`GB_is_fmin_monoid'', `1'')\n') ;
    end
else
    % not a min monoid
    fprintf (f, 'define(`GB_is_imin_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_is_fmin_monoid'', `0'')\n') ;
end

% max monoids:
if (is_max)
    if (contains (ztype, 'int'))
        % max monoid for signed or unsigned integers
        fprintf (f, 'define(`GB_is_imax_monoid'', `1'')\n') ;
        fprintf (f, 'define(`GB_is_fmax_monoid'', `0'')\n') ;
    else
        % max monoid for float or double
        fprintf (f, 'define(`GB_is_imax_monoid'', `0'')\n') ;
        fprintf (f, 'define(`GB_is_fmax_monoid'', `1'')\n') ;
    end
else
    % not a max monoid
    fprintf (f, 'define(`GB_is_imax_monoid'', `0'')\n') ;
    fprintf (f, 'define(`GB_is_fmax_monoid'', `0'')\n') ;
end

% only PLUS, TIMES, LOR, LAND, and LXOR can be done with OpenMP atomics
% in gcc and icc.  However, only PLUS and TIMES work with OpenMP atomics
% in Microsoft Visual Studio; the LOR, LAND, and LXOR atomics don't compile.
fprintf (f, 'define(`GB_has_omp_atomic'', `%d'')\n', omp_atomic) ;
fprintf (f, 'define(`GB_microsoft_has_omp_atomic'', `%d'')\n', omp_microsoft_atomic) ;

% to get an entry from A
if (is_second || is_pair || is_positional)
    % value of A is ignored for the SECOND and PAIR operators
    fprintf (f, 'define(`GB_geta'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_geta'', `%s $1 = $2 [$3]'')\n', xytype) ;
end

% to get an entry from B
if (is_first || is_pair || is_positional)
    % value of B is ignored for the FIRST and PAIR operators
    fprintf (f, 'define(`GB_getb'', `;'')\n') ;
else
    fprintf (f, 'define(`GB_getb'', `%s $1 = $2 [$3]'')\n', xytype) ;
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

% create the multiply operator (assignment)
mult2 = strrep (mult,  'xarg', '`$2''') ;
mult2 = strrep (mult2, 'yarg', '`$3''') ;
fprintf (f, 'define(`GB_multiply'', `$1 = %s'')\n', mult2) ;

% create the add update, of the form w += t
if (is_min)
    if (contains (ztype, 'int'))
        % min monoid for signed or unsigned integers
        add2 = 'if ($1 > $2) $1 = $2' ;
    else
        % min monoid for float or double, with omitnan property
        add2 = 'if (!isnan ($2) && !islessequal ($1, $2)) $1 = $2' ;
    end
elseif (is_max)
    if (contains (ztype, 'int'))
        % max monoid for signed or unsigned integers
        add2 = 'if ($1 < $2) $1 = $2' ;
    else
        % max monoid for float or double, with omitnan property
        add2 = 'if (!isnan ($2) && !isgreaterequal ($1, $2)) $1 = $2' ;
    end
else
    % use the add function as given
    add2 = strrep (add,  'w', '`$1''') ;
    add2 = strrep (add2, 't', '`$2''') ;
end
fprintf (f, 'define(`GB_add_update'', `%s'')\n', add2) ;

% create the add function, of the form w + t
add2 = strrep (addfunc,  'w', '`$1''') ;
add2 = strrep (add2,     't', '`$2''') ;
fprintf (f, 'define(`GB_add_function'', `%s'')\n', add2) ;

% create the multiply-add operator
is_imin_or_imax = (isequal (addop, 'min') || isequal (addop, 'max')) && contains (ztype, 'int') ;

if (~is_imin_or_imax && ...
    (isequal (ztype, 'float') || isequal (ztype, 'double') || ...
     isequal (ztype, 'bool') || is_first || is_second || is_pair || is_positional))
    % float and double do not get promoted.
    % bool is OK since promotion of the result (0 or 1) to int is safe.
    % first and second are OK since no promotion occurs.
    % is* operators are OK too.
    multadd = strrep (add, 't',  mult) ;
    multadd = strrep (multadd, 'w', '`$1''') ;
    multadd = strrep (multadd, 'xarg', '`$2''') ;
    multadd = strrep (multadd, 'yarg', '`$3''') ;
    fprintf (f, 'define(`GB_multiply_add'', `%s'')\n', multadd) ;
else
    % use explicit typecasting to avoid ANSI C integer promotion.
    add2 = strrep (add,  'w', '`$1''') ;
    add2 = strrep (add2, 't', 'x_op_y') ;
    fprintf (f, 'define(`GB_multiply_add'', `%s x_op_y = %s ; %s'')\n', ...
        ztype, mult2, add2) ;
end

% create the disable flag
disable  = sprintf ('GxB_NO_%s', upper (addop)) ;
if (~isequal (addop, multop))
    disable = [disable (sprintf (' || GxB_NO_%s', upper (multop)))] ;
end
disable = [disable (sprintf (' || GxB_NO_%s', upper (fname)))] ;
disable = [disable (sprintf (' || GxB_NO_%s_%s', upper (addop), upper (zname)))] ;
if (~ (isequal (addop, multop) && isequal (zname, fname)))
    disable = [disable (sprintf (' || GxB_NO_%s_%s', upper (multop), upper (fname)))] ;
end
disable = [disable (sprintf (' || GxB_NO_%s_%s_%s', ...
    upper (addop), upper (multop), upper (fname))) ] ;
fprintf (f, 'define(`GB_disable'', `(%s)'')\n', disable) ;
fclose (f) ;

% construct the *.c file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.c | m4 | tail -n +39 > Generated/GB_AxB__%s.c', ...
name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
cmd = sprintf (...
'cat control.m4 Generator/GB_AxB.h | m4 | tail -n +39 >> Generated/GB_AxB__include.h') ;
system (cmd) ;

delete ('control.m4') ;

