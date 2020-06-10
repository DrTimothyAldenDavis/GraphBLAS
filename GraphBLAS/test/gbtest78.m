function gbtest78
%GBTEST78 test integer operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

A = uint8 (magic (4)) ;
A = A (:,1:3) ;
A (1,1) = 0 ;

assert (GrB.isbycol (A)) ;
assert (~GrB.isbyrow (A)) ;

disp (A, GrB (5)) ;

G = GrB (A) ;

C = (G < -1) ;
assert (isequal (C, sparse (false (4,3)))) ;

C = (-1 < G) ;
assert (isequal (C, sparse (true (4,3)))) ;

C = GrB.empty ;
assert (isequal (C, [ ])) ;

C1 = bitset (A, 1, 1) ;
C2 = bitset (G, 1, GrB (1)) ;
assert (isequal (C1, C2)) ;

C1 = bitshift (uint64 (3), A) ;
C2 = bitshift (uint64 (3), G) ;
assert (isequal (C1, C2)) ;

fprintf ('gbtest78: all tests passed\n') ;

