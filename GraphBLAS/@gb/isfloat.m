function s = isfloat (G)
%ISFLOAT true for floating-point GraphBLAS matrices.
%
% See also isnumeric, isreal, isinteger, islogical, gb.type, isa, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

t = gbtype (G.opaque) ;
s = isequal (t, 'double') || isequal (t, 'single') || isequal (t, 'complex') ;

