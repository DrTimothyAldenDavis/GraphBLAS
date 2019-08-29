function s = islogical (G)
%ISINTEGER true for logical GraphBLAS matrices.
%
% See also isnumeric, isfloat, isreal, isinteger, gb.type, isa, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = isequal (gbtype (G.opaque), 'logical') ;

