function C = power (A, B)
%.^ Array power.
% C = A.^B computes element-wise powers.  One or both of A and B may be
% scalars.  Otherwise, A and B must have the same size.  C is sparse (with
% the same patternas A) if B is a positive scalar (greater than zero), or
% full otherwise.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB/mpower.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% ensure A and B are both GrB matrices
if (~isa (A, 'GrB'))
    A = GrB (A) ;
end
if (~isa (B, 'GrB'))
    B = GrB (B) ;
end

% determine if C = A.^B is real or complex
if (isreal (A) && isreal (B))
    % A and B are both real.  Check the contents of B.
    if (islogical (B) || isinteger (B) || isequal (B, round (B)))
        % All entries in B are integers, so C is real
        c_is_real = true ;
    elseif (GrB.reduce ('min', A) >= 0)
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

atype = GrB.type (A) ;
btype = GrB.type (B) ;

if (c_is_real)
    % C has the same type as A
    ctype = GrB.optype (atype, btype) ;
else
    % C is complex
    if (contains (GrB.type (A), 'single') && ...
        contains (GrB.type (B), 'single'))
        ctype = 'single complex' ;
    else
        ctype = 'double complex' ;
    end
end

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars
        A = gb_full (A, ctype) ;
        B = gb_full (B, ctype) ;
    else
        % A is a scalar, B is a matrix; expand A to the size of B
        [m, n] = size (B) ;
        % A (1:m,1:n) = A, converting to ctype
        A = GrB.subassign (GrB (m, n, ctype), A) ;
        B = gb_full (B, ctype) ;
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        if (gb_get_scalar (B) <= 0)
            % so the result is full
            A = gb_full (A, ctype) ;
            [m, n] = size (A) ;
            % B (1:m,1:n) = B, converting to ctype
            B = GrB.subassign (GrB (m, n, ctype), B) ;
        else
            % The scalar b is > 0, and thus 0.^b is zero.  The result is
            % sparse.  B is expanded to a matrix with the same pattern as
            % A, with the type of C.
            B = GrB.expand (GrB (B, ctype), A) ;
        end
    else
        % both A and B are matrices.
        A = gb_full (A, ctype) ;
        B = gb_full (B, ctype) ;
    end
end

% C = A.^B, where A and B now have the same pattern
C = GrB.emult ('pow', A, B) ;

% convert C back to real, if complex but with all-zero imaginary part
if (~c_is_real && nnz (imag (C)) == 0)
    C = real (C) ;
end

