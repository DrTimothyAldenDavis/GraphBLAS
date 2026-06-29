function gbtest50 (ghb)
%GBTEST50 test [GrB,GhB].ktruss and [GrB,GhB].tricount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

% The MathWorks has mangled the Harwell-Boeing west0479 matrix,
% by reducing the precision of its entries, and dropping one entry.
% The correct version is in the HB/west0479 matrix at sparse.tamu.edu.

load west0479_correct.txt
west0479 = spconvert (west0479_correct) ;

A = gtb_offdiag (ghb, west0479) ;
A = A+A' ;
C3a  = gtb_ktruss (ghb, A) ;
C3  = gtb_ktruss (ghb, A, 3) ;
assert (isequal (C3a, C3)) ;
C3  = gtb_ktruss (ghb, A, 3, 'check') ;
assert (isequal (C3a, C3)) ;

ntriangles = sum (C3, 'all') / 6 ;
assert (ntriangles == 237) ;

C4a = gtb_ktruss (ghb, A, 4) ;
C4b = gtb_ktruss (ghb, C3, 4) ;          % this is faster
assert (isequal (C4a, C4b)) ;

nt2 = gtb_tricount (ghb, A) ;
assert (ntriangles == nt2) ;

d = gtb_entries (ghb, A, 'col', 'degree') ;
nt2 = gtb_tricount (ghb, A, d) ;
assert (ntriangles == nt2) ;

nt2 = gtb_tricount (ghb, A, 'check', d) ;
assert (ntriangles == nt2) ;

nt2 = gtb_tricount (ghb, A, d, 'check') ;
assert (ntriangles == nt2) ;

rng ('default') ;
for k = 1:200
    if (mod (k, 10) == 1)
        fprintf ('.') ;
    end
    n = 10000 ;
    G = gtb_eye (ghb, 10000) ;
    j = randperm (n, 10) ;
    G (:,j) = 1 ;
    G (j,:) = 1 ;
    nt = gtb_tricount (ghb, G) ; %#ok<NASGU>
end

fprintf ('\ngbtest50 (%d): all tests passed\n', ghb) ;

