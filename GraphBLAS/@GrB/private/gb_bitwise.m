function C = gb_bitwise (op, A_arg, B_arg, assumedtype)
%GB_BITWISE bitwise AND, OR, XOR, ...

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

atype = gbtype (A_arg) ;
btype = gbtype (B_arg) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical') || isequal (btype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

if (~gb_contains (assumedtype, 'int'))
    error ('GrB:error', 'assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

if (isequal (atype, 'double') || isequal (atype, 'single'))
    A = gbnew (A_arg, assumedtype) ;
    atype = assumedtype ;
else
    A = A_arg ;
end

if (isequal (op, 'bitshift'))

    if (~isequal (btype, 'int8'))
        % convert B to int8, and ensure all values are in range -64:64
        % ensure all entries in B are <= 64
        B = gbapply2 (['min.' btype], B_arg, 64) ;
        if (gb_issigned (btype))
            % ensure all entries in B are >= -64
            B = gbapply2 (['max.' btype], B, -64) ;
        end
        B = gbnew (B, 'int8') ;
    else
        B = B_arg ;
    end

    a_is_scalar = gb_isscalar (A) ;
    b_is_scalar = gb_isscalar (B) ;

    if (a_is_scalar && ~b_is_scalar)
        % A is a scalar, B is a matrix
        C = gbapply2 (['bitshift.' atype], gbfull (A), B) ;
    elseif (~a_is_scalar && b_is_scalar)
        % A is a matrix, B is a scalar
        C = gbapply2 (['bitshift.' atype], A, gbfull (B)) ;
    else
        % both A and B are matrices, or both are scalars
        % expand B by padding it with zeros from the pattern of A
        B = gbeadd ('1st.int8', B, gb_expand (0, A, 'int8')) ;
        C = gbemult (['bitshift.' atype], A, B) ;
    end

else

    if (isequal (btype, 'double') || isequal (btype, 'single'))
        B = gbnew (B_arg, assumedtype) ;
        btype = assumedtype ;
    else
        B = B_arg ;
    end
    if (~isequal (atype, btype))
        error ('GrB:error', 'integer inputs must have the same type') ;
    end

    switch (op)
        case { 'bitxor', 'bitor' }
            C = gb_eadd (A, op, B) ;
        case { 'bitand' }
            C = gb_emult (A, op, B) ;
    end
end

if (~isequal (gbtype (C), ctype))
    C = gbnew (C, ctype) ;
end

