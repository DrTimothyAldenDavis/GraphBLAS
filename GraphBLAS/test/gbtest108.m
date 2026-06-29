function gbtest108 (ghb)
%GBTEST108 test mat2cell

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

S = magic (5) ;
A = gtb (ghb, S) ;
C1 = mat2cell (A, [2 3]) ;
C2 = mat2cell (S, [2 3]) ;
assert (isequal (C1, C2)) ;
C3 = mat2cell (S, gtb (ghb, [2 3])) ;
assert (isequal (C1, C3)) ;

fprintf ('\ngbtest108 (%d): all tests passed\n', ghb) ;

