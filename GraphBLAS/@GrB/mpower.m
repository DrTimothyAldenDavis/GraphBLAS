function C = mpower (A, B)
%A^B Matrix power.
% C = A^B computes the matrix power of A raised to the B. A must be a
% square matrix.  B must an integer >= 0.
%
% The inputs may be either GraphBLAS and/or MATLAB matrices/scalars, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB/power.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO

if (isscalar (A) && isscalar (B))
    C = power (A, B) ;
    return ;
end

[m, n] = size (A) ;

if (m ~= n)
    gb_error ('For C=A^B, A must be square') ;
end

if (~isscalar (B))
    gb_error ('For C=A^B, B must be a non-negative integer scalar') ;
end

b = gb_get_scalar (B) ;
if (isreal (b) && isfinite (b) && round (b) == b && b >= 0)
    if (b == 0)
        type = GrB.type (A) ;
        if (isequal (type, 'single complex'))
            C = GrB.eye (n, 'single') ;
        elseif (isequal (type, 'double complex'))
            C = GrB.eye (n, 'double') ;
        else
            % C is identity, of the same type as A
            C = GrB.eye (n, type) ;
        end
    else
        % C = A^b where b > 0 is an integer
        C = gb_compute_mpower (A, b) ;
    end
else
    gb_error ('For C=A^B, B must be a non-negative integer scalar') ;
end

end

function C = gb_compute_mpower (A, b)
% C = A^b where b > 0 is an integer
if (b == 1)
    C = A ;
else
    T = gb_compute_mpower (A, floor (b/2)) ;
    C = T*T ;
    clear T ;
    if (mod (b, 2) == 1)
        C = C*A ;
    end
end
end

