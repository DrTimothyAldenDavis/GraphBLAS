function C = floor (G)
%FLOOR round entries to nearest integers towards -infinity.
% C = floor (G) rounds the entries in the GraphBLAS matrix G to the
% nearest integers towards -infinity.
%
% See also GrB/ceil, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isfloat (G) && GrB.entries (G) > 0)
    C = GrB.apply ('floor', G) ;
else
    C = G ;
end

