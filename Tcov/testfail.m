function testfail (threads,longtests)
%TESTFAIL run a few GraphBLAS tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_init ;

testall_time = tic ;

if (nargin < 2)
    % run the shorter tests by default
    longtests = 0 ;
end

if (nargin < 1)
    threads = [ ] ;
end
if (isempty (threads))
    threads {1} = [4 1] ;
end
t = threads ;

% single thread
s {1} = [1 1] ;

extra {1} = [4 1] ;
extra {2} = [1 1] ;

% clear the statement coverage counts
clear global GraphBLAS_grbcov

% use built-in complex data types by default
GB_builtin_complex_set (true) ;

% many of the tests use spok in SuiteSparse, a copy of which is
% included here in GraphBLAS/Test/spok.
addpath ('../Test/spok') ;

try
    spok (sparse (1)) ;
catch
    here = pwd ;
    cd ../Test/spok ;
    spok_install ;
    cd (here) ;
end

logstat ;             % start the log.txt
hack = GB_mex_hack ;

% JIT controls
clear o f b
o = {-4} ;      % reset on
f = {0} ;       % off
b = {-4, 0} ;   % reset on; off
% o = b ;

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;

malloc_debugging = stat ;

%===============================================================================
% quick tests for statement coverage, with malloc debugging
%===============================================================================

% Timings below are for test coverage (Tcov), with malloc debuging enabled, on
% hypersparse.cse.tamu.edu (20 core Xeon).  Times will differ if this test is
% run with malloc debugging off.

%----------------------------------------
% tests with high rates (over 100/sec)
%----------------------------------------

% dir
logstat ('test253',t,b) ; % basic JIT tests
% dir
% logstat ('test252',t,b) ; % basic tests
% dir
% logstat ('test251',t,b) ; % dot4, dot2, with plus_pair
% dir
% logstat ('test250',t,o) ; % basic tests
% dir

