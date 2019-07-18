function testall (threads,longtests)
%TESTALL run all GraphBLAS tests
%
% Usage:
% testall ;         % runs just the shorter tests (about 15 minutes)
% testall([ ],1) ;  % runs all the tests (overnight).  Requires SuiteSparse.
%
% testall(threads) ;    % run with specific list of threads and chunk sizes
%
% threads is a cell array. Each entry is 2-by-1, with the first value being
% the # of threads to use and the 2nd being the chunk size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

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
    % threads {2} = [1 4096] ;
end
t = threads ;

% single thread
t1 {1} = [1 1] ;

s {1} = [4 1] ;
s {2} = [1 1] ;

% clear the statement coverage counts
clear global GraphBLAS_gbcov

% many of the tests use SuiteSparse/MATLAB_Tools/spok, a copy of which is
% included here in GraphBLAS/Test/spok.
addpath ('../Test/spok') ;
addpath ('../Demo/MATLAB') ;

try
    spok (sparse (1)) ;
catch
    cd spok ; spok_install ; cd ..
end

logstat ;             % start the log.txt

%-------------------------------------------------------------------------------
% quick tests for statement coverage
%-------------------------------------------------------------------------------

% Timings below are for test coverage (Tcov), with malloc debuging enabled.
% Times will differ if this test is run in with malloc debugging off.

%----------------------------------------
% test taking less than 1 second:
%----------------------------------------

%0.
logstat ('test72',t) ;  % several special cases
logstat ('test72',t1) ; % several special cases
%0
logstat ('test07',t) ;  % quick test GB_mex_subassign
logstat ('test07',t1) ; % quick test GB_mex_subassign
%0
logstat ('test07b',t) ; % quick test GB_mex_assign
%0
logstat ('test09',t) ;  % duplicate I,J test of GB_mex_subassign
%0
logstat ('test83',t) ;  % GrB_assign with C_replace and empty J
%0
logstat ('test84',t) ;  % GrB_assign (row and column with C in CSR format)
logstat ('test84',t1) ; % GrB_assign (row and column with C in CSR format)
%0
logstat ('test85',t) ;  % GrB_transpose (1-by-n with typecasting)
logstat ('test85',t1) ; % GrB_transpose (1-by-n with typecasting)
%0.1
logstat ('test109',t) ; % terminal monoid with user-defined type
logstat ('test109',t1); % terminal monoid with user-defined type
%0.1
logstat ('test110',t) ; % binary search of M(:,j) in accum/mask
%0.1
logstat ('test98',t) ;  % GB_mex_mxm, typecast on the fly
logstat ('test98',t1);  % GB_mex_mxm, typecast on the fly
%0.1
logstat ('test92',t) ;  % GB_subref (symbolic case)
%0.1
logstat ('test97',t) ;  % GB_mex_assign, scalar expansion and zombies
%0.1
logstat ('test01',t) ;  % error handling
logstat ('test01',t1) ; % error handling
%0.1
logstat ('test04',t) ;  % simple mask and transpose test
%0.1
logstat ('test05',t) ;  % quick setElement test, with typecasting
logstat ('test05',t1);  % quick setElement test, with typecasting
%0.1
logstat ('test15',t) ;  % simple test of GB_mex_AxB
logstat ('test15',t1) ; % simple test of GB_mex_AxB
%0.1
logstat ('test78',t) ;  % quick test of hypersparse subref
%0.1
logstat ('test82',t) ;  % GrB_extract with index range (hypersparse)
%0.1
logstat ('test94',t) ;  % pagerank
logstat ('test94',t1) ; % pagerank
%0.2
logstat ('test128',t) ; % eWiseMult, different cases in emult_phase0
%0.2
logstat ('test126',t) ; % test GrB_reduce to vector on a very sparse matrix 
%0.2
logstat ('test03',t) ;  % random matrix tests
logstat ('test03',t1) ; % random matrix tests
%0.4
logstat ('test02',t) ;  % matrix copy and dup tests
%0.4
logstat ('test17',t) ;  % quick test of GrB_*_extractElement
%0.5
logstat ('test108',t) ; % boolean monoids
%0.6
logstat ('test124',t) ; % GrB_extract, case 6
%0.6
logstat ('test101',t) ; % GrB_*_import and export
%0.6
logstat ('test26') ;    % quick test of GxB_select

%----------------------------------------
% tests taking 1 to 10 seconds:
%----------------------------------------

%1
logstat ('test80',t) ;  % test GrB_mxm on all semirings (different matrix)
%1.1
logstat ('test104',t) ; % export/import
%1.2
logstat ('test102',t);  % GB_AxB_flopcount
%2
logstat ('test12',t) ;  % Wathen finite-element matrices (short test)
%2.3
logstat ('test105',t) ; % eWiseAdd for hypersparse
%2.6
logstat ('test29',t) ;  % reduce with zombies
%2.9
logstat ('test107') ;   % monoids with terminal values
%4
logstat ('test100',t) ; % GB_mex_isequal
logstat ('test100',t1); % GB_mex_isequal
%4.9
logstat ('test30') ;    % performance test GB_mex_subassign, scalar expansion
%5
logstat ('test93',t) ;  % pagerank
%5.1
logstat ('test11',t) ;  % exhaustive test of GrB_extractTuples
%7
logstat ('test28',t) ;  % mxm with aliased inputs, C<C> = accum(C,C*C)
%10 + 5
logstat ('test19b',t) ; % GrB_assign, many pending operators (malloc debug off)
logstat ('test19b',t1); % GrB_assign, many pending operators (malloc debug off)
%8
logstat ('test103',t) ; % GrB_transpose aliases

%----------------------------------------
% tests taking 10 to 200 seconds
%----------------------------------------

%13
logstat ('test14',t) ;  % GrB_reduce
%25
logstat ('test76',t) ;  % GxB_resize
%26
logstat ('test125',t) ; % test GrB_mxm: row and column scaling
%28
logstat ('test23',t) ;  % quick test of GB_*_build
%28
logstat ('test106',t) ; % GxB_subassign with alias
%29
logstat ('test77',t) ;  % quick tests of GxB_kron
%31 + 4
logstat ('test00',t) ;  % GB_mex_mis
logstat ('test00',t1);  % GB_mex_mis
%31
logstat ('test88',t) ;  % hypersparse matrices with heap-based method
%39
logstat ('test69',t) ;  % assign and subassign with alias
%48
logstat ('test99',t) ;  % GB_mex_transpose with explicit zeros in the Mask
%64
logstat ('test54',t) ;  % assign and extract with begin:inc:end
%80
logstat ('test27',t) ;  % quick test of GxB_select (band)
%82
logstat ('test127',t) ; % test eWiseAdd, eWiseMult (all types and operators)
%92
logstat ('test39(0)') ; % tests of GrB_transpose, GB_*_add and eWiseAdd

%122
logstat ('test53',t) ;  % quick test of GB_mex_Matrix_extract

%166+5
logstat ('test19',t) ;  % GxB_subassign, many pending operators
% logstat ('test19',t1) ; % GxB_subassign, many pending operators

%191 no debug, 533 with
logstat ('test10',t) ;  % GrB_apply

%----------------------------------------
% longer tests (200 seconds to 600 seconds)
%----------------------------------------

% TODO HERE

%245
logstat ('test16',t) ;  % user-defined complex operators
%268
logstat ('test81',s) ;  % GrB_Matrix_extract with stride, range, backwards
%334
logstat ('test45(0)',s) ;  % test GB_mex_setElement and build
%337
logstat ('test75',s) ;  % test GrB_mxm A'*B on all semirings
%439
logstat ('test20',s) ;  % quick test of GB_mex_mxm on a few semirings
%495
logstat ('test74',t) ;  % test GrB_mxm on all semirings
%512
logstat ('test90',s) ;  % test pre-compiled user-defined semirings
%532
logstat ('test06',s) ;  % test GrB_mxm on all semirings

%----------------------------------------
% too long (more than 600 seconds)
%----------------------------------------

% TODO reduce the length of these tests

%113 no malloc debug, 940 with (TODO: try turning off debub)
logstat ('test21',t) ;  % quick test of GB_mex_subassign
logstat ('test21',t1) ;  % quick test of GB_mex_subassign

%1292
logstat ('test25',s) ;  % quick test of GxB_select

%115 no debug, 2085 with (TODO: try turning off debug)
logstat ('test21b',t) ; % quick test of GB_mex_assign

%2897
logstat ('test18',s) ;  % quick tests of GrB_eWiseAdd and eWiseMult

%-------------------------------------------------------------------------------
% The following tests are not required for statement coverage.  Some need
% other packages in SuiteSparse (CSparse, SSMULT, ssget).  By default, these
% tests are not run.  To install them, see test_other.m.

if (longtests)
    % useful tests but not needed for statement coverage
    logstat ('test89',t) ;     % performance test of complex A*B
    logstat ('test13',t) ;     % simple tests of GB_mex_transpose
    logstat ('test22',t) ;     % quick test of GB_mex_transpose
    logstat ('test26(1)',t) ;  % longer test of GxB_select
    logstat ('test20(1)',t) ;  % test of GB_mex_mxm on all built-in semirings
    logstat ('test18(1)',t) ;  % lengthy tests of GrB_eWiseAdd and eWiseMult
    logstat ('test08b',t) ;    % quick test GB_mex_assign
    logstat ('test09b',t) ;    % duplicate I,J test of GB_mex_assign
    logstat ('test21(1)',t) ;  % exhaustive test of GB_mex_subassign
    logstat ('test23(1)',t) ;  % exhaustive test of GB_*_build
    logstat ('test24(1)',t) ;  % exhaustive test of GrB_Matrix_reduce
    logstat ('test64',t) ;     % quick test of GB_mex_subassign, scalar expan.
    logstat ('test65',t) ;     % type casting
    logstat ('test66',t) ;     % quick test for GrB_Matrix_reduce
    logstat ('test67',t) ;     % quick test for GrB_apply
    logstat ('test31',t) ;     % simple tests of GB_mex_transpose
    logstat ('test12(0)',t) ;  % Wathen finite-element matrices (full test)
    logstat ('test58(0)') ;    % longer GB_mex_eWiseAdd_Matrix performance test
    logstat ('test32',t) ;     % quick GB_mex_mxm test
    logstat ('test33',t) ;     % create a semiring
    logstat ('test34',t) ;     % quick GB_mex_eWiseAdd_Matrix test
    logstat ('test35') ;       % performance test for GrB_extractTuples
    logstat ('test36') ;       % performance test for GB_mex_Matrix_subref
    logstat ('test37') ;       % performance test for GrB_qsort1
    logstat ('test38',t) ;     % GB_mex_transpose with matrix collection
    logstat ('test39') ;       % tests of GrB_transpose, GB_*_add and eWiseAdd
    logstat ('test40',t) ;     % test for GrB_Matrix_extractElement, and Vector
    logstat ('test41',t) ;     % test of GB_mex_AxB
    logstat ('test42') ;       % performance tests for GB_mex_Matrix_build
    logstat ('test43',t) ;     % performance tests for GB_mex_Matrix_subref
    logstat ('test44',t) ;     % test qsort
    logstat ('test53',t) ;     % exhaustive test of GB_mex_Matrix_extract
    logstat ('test62',t) ;     % exhaustive test of GrB_apply
    logstat ('test63',t) ;     % GB_mex_op and operator tests
    logstat ('test46') ;       % performance test GB_mex_subassign
    logstat ('test46b') ;      % performance test GB_mex_assign
    logstat ('test47',t) ;
    logstat ('test48') ;  
    logstat ('test49') ;  
    logstat ('test50',t) ;     % test GB_mex_AxB on larger matrix
    logstat ('test51') ;       % performance test GB_mex_subassign, multiple ops
    logstat ('test51b') ;      % performance test GB_mex_assign, multiple ops
    logstat ('test52',t) ;     % performance of A*B with tall mtx, AdotB, AxB
    logstat ('test06(936)',t); % performance test of GrB_mxm on all semirings
    logstat ('test55',t) ;
    logstat ('test55b',t) ;
    logstat ('test56',t) ;
    logstat ('test57',t) ;
    logstat ('test58') ;  
    logstat ('test59',t) ;
    logstat ('test60',t) ;
    logstat ('test61') ;  

    logstat ('test64b',t) ;
    logstat ('test68',t) ;
    logstat ('test70',t) ;
    logstat ('test71',t) ;
    logstat ('test73',t) ;
    logstat ('test79',t) ;
    logstat ('test86',t) ;
    logstat ('test87',t) ;
    logstat ('test91',t) ;
    logstat ('test93b',t) ;
    logstat ('test95',t) ;

    logstat ('test111',t) ;
    logstat ('test112',t) ;
    logstat ('test113',t) ;
    logstat ('test114',t) ;
    logstat ('test116',t) ;
    logstat ('test117',t) ;
    logstat ('test118',t) ;
    logstat ('test119',t) ;
    logstat ('test120',t) ;
    logstat ('test121',t) ;
    logstat ('test122',t) ;
    logstat ('test123',t) ;

    % tested via test16:
    logstat ('testc1',t) ;
    logstat ('testc2',t) ;
    logstat ('testc3',t) ;
    logstat ('testc4',t) ;
    logstat ('testc5',t) ;
    logstat ('testc6',t) ;
    logstat ('testc7',t) ;
    logstat ('testc8',t) ;
    logstat ('testc9',t) ;
    logstat ('testca',t) ;
    logstat ('testcb',t) ;
    logstat ('testcc',t) ;

    %8.5
    logstat ('test30b') ;   % performance test GB_mex_assign, scalar expansion
    %16
    logstat ('test96',t) ;  % A*B using dot product
    %42
    logstat ('test24',t) ;  % test of GrB_Matrix_reduce
    %10
    logstat ('test115',t) ; % GrB_assign with duplicate indices
    %35
    logstat ('test08',t) ;  % quick test GB_mex_subassign

end

t = toc (testall_time) ;
fprintf ('\ntestall: all tests passed, total time %g sec\n', t) ;

