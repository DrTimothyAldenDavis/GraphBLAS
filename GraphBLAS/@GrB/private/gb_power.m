function C = gb_power (A, B)
%GB_POWER .^ Array power.
% C = A.^B computes element-wise powers.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
a_is_real = ~contains (atype, 'complex') ;
b_is_real = ~contains (btype, 'complex') ;

% determine if C = A.^B is real or complex
if (a_is_real && b_is_real)
    % A and B are both real.  Determine if C might be complex.
    if (contains (btype, 'int') || isequal (btype, 'logical'))
        % B is logical or integer, so C is real
        c_is_real = true ;
    elseif (gbisequal (B, gbapply ('round', B)))
        % B is floating point, but all values are equal to integers
        c_is_real = true ;
    elseif (gb_scalar (gbreduce ('min', A)) >= 0)
        % All entries in A are non-negative, so C is real
        c_is_real = true ;
    else
        % A contains negative entries, and B is non-integer, so C can
        % be complex.
        c_is_real = false ;
    end
else
    % A or B are complex, or both, so C must be complex
    c_is_real = false ;
end

if (c_is_real)
    % C is real
    ctype = gboptype (atype, btype) ;
else
    % C is complex
    if (contains (atype, 'single') && contains (btype, 'single'))
        ctype = 'single complex' ;
    else
        ctype = 'double complex' ;
    end
end

if (a_is_scalar)

    %----------------------------------------------------------------------
    % A is a scalar: C is a full matrix
    %----------------------------------------------------------------------

    if (b_is_scalar)
        % both A and B are scalars
        A = gbfull (A, ctype) ;
    else
        % A is a scalar, B is a matrix; expand A to the size of B
        A = gb_scalar_to_full (bm, bn, ctype, A) ;
    end
    B = gbfull (B, ctype) ;

else

    %----------------------------------------------------------------------
    % A is a matrix
    %----------------------------------------------------------------------

    if (b_is_scalar)
        % A is a matrix, B is a scalar
        b = gb_scalar (B) ;
        if (b == 0)
            % special case:  C = A.^0 = ones (am, an, ctype)
            C = gb_scalar_to_full (am, an, ctype, 1) ;
            return ;
        elseif (b == 1)
            % special case: C = A.^1 = A
            C = A ;
            return
        elseif (b == 2)
            % special case: C = A.^2
            C = gbemult (A, '*', A) ;
            return ;
        elseif (b <= 0)
            % 0.^b where b < 0 is Inf, so C is full
            A = gbfull (A, ctype) ;
            B = gb_scalar_to_full (am, an, ctype, B) ;
        else
            % The scalar b is > 0, and thus 0.^b is zero.  The result is
            % sparse.  B is expanded to a matrix with the same pattern as
            % A, with the type of C.
            B = gb_expand (B, A, ctype) ;
        end
    else
        % both A and B are matrices.
        A = gbfull (A, ctype) ;
        B = gbfull (B, ctype) ;
    end

end

% C = A.^B, where A and B now have the same pattern
if (c_is_real)
    C = gbemult ('pow', A, B) ;
else
    C = gb_to_real_if_imag_zero (gbemult ('pow', A, B)) ;
end

