function gbtest123 (ghb)
%GBTEST123 test build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 1000 ;
H = gtb (ghb, n, n) ;
H (1,1) = 1 ;
S = gtb_build (ghb, H,H,pi) ;
P = sparse (pi) ;
assert (isequal (S, P)) ;

n = flintmax ;
H = gtb (ghb, n, n) ;
H (1,1) = 1 ;
try
    S = gtb_build (ghb, H,H,H) ;
    ok = false ;
catch expected_error
    expected_error
    have_octave = gb_octave ;
    if (have_octave)
        assert (isequal (expected_error.message, ...
            'gbmex_build: input matrix dimensions are too large')) ;
    else
        assert (isequal (expected_error.message, ...
            'input matrix dimensions are too large')) ;
    end
    ok = true ;
end
assert (ok) ;

fprintf ('\ngbtest123 (%d): all tests passed\n', ghb) ;

