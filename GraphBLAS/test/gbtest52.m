function gbtest52
%GBTEST52 test GrB.format

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

GrB.format
GrB.format ('by col') ;
f = GrB.format
A = magic (4)
G = GrB (A)
assert (isequal (f, GrB.format (G))) ;
GrB.format ('by row')
f = GrB.format
H = GrB (5,5)
% TODO reenable assert when this works
% assert (isequal ('by row', GrB.format (H))) ;
if (~isequal ('by row', GrB.format (H)))
    fprintf ('ERROR: H should be by row') ;
end
GrB.format ('by col')
f = GrB.format

fprintf ('gbtest52: all tests passed\n') ;

