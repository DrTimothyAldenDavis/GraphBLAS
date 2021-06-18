function test93b
%TEST93B test dpagerank and ipagerank

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
addpath ('../Test') ;
addpath ('../Test/spok') ;
addpath ('../Demo/MATLAB') ;

n = 10 ;

    fprintf ('\n--------------n: %d\n', n) ;
    nz = 8*n ;
    d = nz / n^2 ;
    A = sprand (n, n, d) ;
    A = spones (A + speye (n)) ;

    tic
    [r1, i1] = dpagerank (A) ;
    t1 = toc ;

    [r2, i2] = GB_mex_dpagerank (A) ;
    t2 = grbresults ;

fprintf ('test93b: all tests passed\n') ;
