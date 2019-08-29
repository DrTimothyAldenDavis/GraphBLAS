function s = isinteger (G)
%ISINTEGER true for integer GraphBLAS matrices.
%
% See also isnumeric, isfloat, isreal, islogical, gb.type, isa, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

t = gbtype (G.opaque) ;
s = isequal (t (1:3), 'int') || isequal (t (1:4), 'uint') ;

