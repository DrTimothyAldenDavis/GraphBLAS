function C = ctranspose (A)
%CTRANSPOSE C = A', matrix transpose a GraphBLAS matrix.
% Note that complex matrices are not yet supported.  When they are, this
% will compute the complex conjugate transpose C=A' when A is complex.
%
% See also gb.gbtranspose, transpose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.gbtranspose (A) ;

