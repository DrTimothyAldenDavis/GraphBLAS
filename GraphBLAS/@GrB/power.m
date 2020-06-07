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

if (isobject (A))
    A = A.opaque ;
end
if (isobject (B))
    B = B.opaque ;
end

C = GrB (gb_power (A, B)) ;

