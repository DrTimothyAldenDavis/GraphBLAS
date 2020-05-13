function C = fix (G)
%FIX Round towards entries in a GraphBLAS matrix to zero.
% C = fix (G) rounds the entries in the GraphBLAS matrix G to the
% nearest integers towards zero.
%
% See also GrB/ceil, GrB/floor, GrB/round.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isfloat (G) && GrB.entries (G) > 0)
    C = GrB.apply ('trunc', G) ;
else
    C = G ;
end

