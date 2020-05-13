function C = pow2 (A, B)
%POW2 base-2 power and scale floating-point number.
% C = pow2 (G) computes C = 2.^G for a GraphBLAS matrix G.
% C = pow2 (F,E) with computes C = F .* (2 .^ fix (E)).
%
% See also GrB/log2, GrB/power.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin == 1)
    % use GrB/power
    C = 2.^G ;
else

    % convert A and B to real
    if (~isreal (A))
        A = real (A) ;
    elseif (~isfloat (A))
        A = GrB (A, 'double') ;
    end
    if (~isreal (B))
        B = real (B) ;
    elseif (~isfloat (B))
        B = GrB (B, 'double') ;
    end

    % use the ldexp operator to compute C = A.*(2.^B)
    C = gb_union_op ('ldexp', A, B) ;
end

