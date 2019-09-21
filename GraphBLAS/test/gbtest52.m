function gbtest52
%GBTEST52 test gb.format

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

gb.format
gb.format ('by col') ;
f = gb.format
A = magic (4)
G = gb (A)
assert (isequal (f, gb.format (G))) ;
gb.format ('by row')
H = gb (5,5)
assert (isequal ('by row', gb.format (H))) ;
gb.format ('by col')

fprintf ('gbtest52: all tests passed\n') ;

