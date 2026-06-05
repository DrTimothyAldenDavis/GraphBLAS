function gbtest131
%GBTEST131 misc error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = GrB (magic (4)) ;

try
    d = diag (A, [1 2]) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (isequal (msg, 'k must be a scalar')) ; 

try
    C = GrB.apply2 (A, '*', [1 2]) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (isequal (msg, 'either A or B must be a non-empty scalar')) ; 

try
    C = GrB.apply2 (A, '*', sparse (0)) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (isequal (msg, 'either A or B must be a non-empty scalar')) ; 

fprintf ('\ngbtest131: all tests passed\n') ;

