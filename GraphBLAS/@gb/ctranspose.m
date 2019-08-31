function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
% Note that complex matrices are not yet supported.  When they are, this
% will compute the complex conjugate transpose C=G' when G is complex.
%
% See also gb.gbtranspose, transpose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.gbtranspose (G) ;

