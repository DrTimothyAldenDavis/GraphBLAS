function C = fix (G)
%FIX Round towards entries in a GraphBLAS matrix to zero.
% C = fix (G) rounds the entries in the GraphBLAS matrix G to the
% nearest integers towards zero.
%
% See also ceil, floor, round.

% FUTURE: this could be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isfloat (G) && gb.nvals (G) > 0)
    [m n] = size (G) ;
    [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
    C = gb.build (i, j, fix (x), m, n) ;
else
    C = G ;
end

