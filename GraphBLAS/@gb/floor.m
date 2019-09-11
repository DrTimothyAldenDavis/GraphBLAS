function C = floor (G)
%FLOOR round entries of a GraphBLAS matrix to the nearest ints to -inf.
% C = floor (G) rounds the entries in the GraphBLAS matrix G to the
% nearest integers towards -infinity.
%
% See also ceil, round, fix.

% FUTURE: this could be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isfloat (G) && gb.entries (G) > 0)
    [m n] = size (G) ;
    [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
    C = gb.build (i, j, floor (x), m, n) ;
else
    C = G ;
end

