function C = ceil (G)
%CEIL round entries of a GraphBLAS matrix to nearest integers towards inf.
%
% See also GrB/floor, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isfloat (G) && GrB.entries (G) > 0)
    C = GrB.apply ('ceil', G) ;
else
    C = G ;
end

