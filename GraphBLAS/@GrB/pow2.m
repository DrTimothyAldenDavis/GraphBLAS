function C = pow2 (A, B)
%POW2 base-2 power and scale floating-point number.
% C = pow2 (G) is C(i,j) = 2.^G(i,j) for each entry in G.
% C = pow2 (F,E) with computes C = F .* (2 .^ fix (E)).
%
% See also GrB/log2, GrB/power.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

if (nargin == 1)

    % use GrB/power
    C = GrB (gb_power (2, A)) ;

else

    if (isobject (B))
        B = B.opaque ;
    end

    type = gboptype (gbtype (A), gbtype (B)) ;

    if (contains (type, 'single'))
        type = 'single' ;
    else
        type = 'double' ;
    end

    % use the ldexp operator to compute C = A.*(2.^B)
    C = GrB (gb_union_op (['ldexp.' type], A, B)) ;

end

