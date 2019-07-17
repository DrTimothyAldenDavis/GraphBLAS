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

logstat ('test74',t) ;  % test GrB_mxm on all semirings
logstat ('test127',t) ; % test eWiseAdd, eWiseMult (all types and operators)
logstat ('test126',t) ; % test GrB_reduce to vector on a very sparse matrix 
logstat ('test30b') ;   % performance test GB_mex_assign, scalar expansion
logstat ('test30') ;    % performance test GB_mex_subassign, scalar expansion
logstat ('test124',t) ; % GrB_extract, case 6
logstat ('test115',t) ; % GrB_assign with duplicate indices
logstat ('test101',t) ; % GrB_*_import and export
logstat ('test103',t) ; % GrB_transpose aliases
logstat ('test104',t) ; % export/import
logstat ('test105',t) ; % eWiseAdd for hypersparse
logstat ('test106',t) ; % GxB_subassign with alias
logstat ('test107') ;   % monoids with terminal values
logstat ('test108',t) ; % boolean monoids
logstat ('test109',t) ; % terminal monoid with user-defined type
logstat ('test110',t) ; % binary search of M(:,j) in accum/mask
logstat ('test98',t) ;  % GB_mex_mxm, typecast on the fly
logstat ('test92',t) ;  % GB_subref (symbolic case)
logstat ('test97',t) ;  % GB_mex_assign, scalar expansion and zombies
logstat ('test100',t) ; % GB_mex_isequal
logstat ('test01',t) ;  % error handling
logstat ('test02',t) ;  % matrix copy and dup tests
logstat ('test03',t) ;  % random matrix tests
logstat ('test04',t) ;  % simple mask and transpose test
logstat ('test05',t) ;  % quick setElement test, with typecasting
logstat ('test07',t) ;  % quick test GB_mex_subassign
logstat ('test07b',t) ; % quick test GB_mex_assign
logstat ('test08',t) ;  % quick test GB_mex_subassign
logstat ('test09',t) ;  % duplicate I,J test of GB_mex_subassign
logstat ('test15',t) ;  % simple test of GB_mex_AxB
logstat ('test17',t) ;  % quick test of GrB_*_extractElement
logstat ('test72',t) ;  % several special cases
logstat ('test26') ;    % quick test of GxB_select
logstat ('test78',t) ;  % quick test of hypersparse subref
logstat ('test12',t) ;  % Wathen finite-element matrices (short test)
logstat ('test54',t) ;  % assign and extract with begin:inc:end
logstat ('test29',t) ;  % reduce with zombies
logstat ('test69',t) ;  % assign and subassign with alias
logstat ('test28',t) ;  % mxm with aliased inputs, C<C> = accum(C,C*C)
logstat ('test11',t) ;  % exhaustive test of GrB_extractTuples
logstat ('test14',t) ;  % GrB_reduce
logstat ('test80',t) ;  % test GrB_mxm on all semirings (different matrix)
logstat ('test81',t) ;  % GrB_Matrix_extract with stride, range, backwards
logstat ('test82',t) ;  % GrB_extract with index range (hypersparse)
logstat ('test83',t) ;  % GrB_assign with C_replace and empty J
logstat ('test84',t) ;  % GrB_assign (row and column with C in CSR format)
logstat ('test85',t) ;  % GrB_transpose (1-by-n with typecasting)
logstat ('test88',t) ;  % hypersparse matrices with heap-based method
logstat ('test00',t) ;  % GB_mex_mis
logstat ('test93',t) ;  % pagerank
logstat ('test94',t) ;  % pagerank
logstat ('test77',t) ;  % quick tests of GxB_kron
logstat ('test76',t) ;  % GxB_resize
logstat ('test102',t);  % GB_AxB_flopcount
logstat ('test27',t) ;  % quick test of GxB_select (band)
logstat ('test125',t) ; % test GrB_mxm: row and column scaling
logstat ('test39(0)') ;  % tests of GrB_transpose, GB_*_add and eWiseAdd
logstat ('test99',t) ;  % GB_mex_transpose with explicit zeros in the Mask
logstat ('test19',t) ;  % GxB_subassign, many pending operators
logstat ('test23',t) ;  % quick test of GB_*_build
logstat ('test96',t) ;  % A*B using dot product
logstat ('test25',t) ;  % quick test of GxB_select
logstat ('test53',t) ;  % quick test of GB_mex_Matrix_extract
logstat ('test24',t) ;  % test of GrB_Matrix_reduce
logstat ('test45(0)',t) ;  % test GB_mex_setElement and build
logstat ('test10',t) ;  % GrB_apply
logstat ('test90',t) ;  % test pre-compiled user-defined semirings
logstat ('test21b',t) ; % quick test of GB_mex_assign
logstat ('test21',t) ;  % quick test of GB_mex_subassign
logstat ('test16',t) ;  % user-defined complex operators
logstat ('test18',t) ;  % quick tests of GrB_eWiseAdd and eWiseMult
logstat ('test75',t) ;  % test GrB_mxm A'*B on all semirings
logstat ('test06',t) ;  % test GrB_mxm on all semirings

% malloc debugging turned off just for this test:
logstat ('test19b',t) ; % GrB_assign, many pending operators

logstat ('test20',t) ;    % quick test of GB_mex_mxm on a few semirings

%-------------------------------------------------------------------------------
% The following tests are not required for statement coverage.  Some need
% other packages in SuiteSparse (CSparse, SSMULT, ssget).  By default, these
% tests are not run.  To install them, see test_other.m.

if (longtests)
    % useful tests but not needed for statement coverage
    logstat ('test89',t) ;  % performance test of complex A*B
    logstat ('test13',t) ;  % simple tests of GB_mex_transpose
    logstat ('test22',t) ;  % quick test of GB_mex_transpose
    logstat ('test26(1)',t) ;  % longer test of GxB_select
    logstat ('test20(1)',t) ;  % test of GB_mex_mxm on all built-in semirings
    logstat ('test18(1)',t) ;  % lengthy tests of GrB_eWiseAdd and eWiseMult
    logstat ('test08b',t) ; % quick test GB_mex_assign
    logstat ('test09b',t) ; % duplicate I,J test of GB_mex_assign
    logstat ('test21(1)',t) ;  % exhaustive test of GB_mex_subassign
    logstat ('test23(1)',t) ;  % exhaustive test of GB_*_build
    logstat ('test24(1)',t) ;  % exhaustive test of GrB_Matrix_reduce
    logstat ('test64',t) ;  % quick test of GB_mex_subassign, scalar expansion
    logstat ('test65',t) ;  % type casting
    logstat ('test66',t) ;  % quick test for GrB_Matrix_reduce
    logstat ('test67',t) ;  % quick test for GrB_apply
    logstat ('test31',t) ;  % simple tests of GB_mex_transpose
    logstat ('test12(0)',t) ; % Wathen finite-element matrices (full test)
    logstat ('test58(0)') ;   % longer GB_mex_eWiseAdd_Matrix performance test
    logstat ('test32',t) ;  % quick GB_mex_mxm test
    logstat ('test33',t) ;  % create a semiring
    logstat ('test34',t) ;  % quick GB_mex_eWiseAdd_Matrix test
    logstat ('test35') ;    % performance test for GrB_extractTuples
    logstat ('test36') ;    % performance test for GB_mex_Matrix_subref
    logstat ('test37') ;    % performance test for GrB_qsort1
    logstat ('test38',t) ;  % GB_mex_transpose with matrix collection
    logstat ('test39') ;    % tests of GrB_transpose, GB_*_add and eWiseAdd
    logstat ('test40',t) ;  % test for GrB_Matrix_extractElement, and Vector
    logstat ('test41',t) ;  % test of GB_mex_AxB
    logstat ('test42') ;    % performance tests for GB_mex_Matrix_build
    logstat ('test43',t) ;  % performance tests for GB_mex_Matrix_subref
    logstat ('test44',t) ;  % test qsort
    logstat ('test53',t) ;  % exhaustive test of GB_mex_Matrix_extract
    logstat ('test62',t) ;  % exhaustive test of GrB_apply
    logstat ('test63',t) ;  % GB_mex_op and operator tests
    logstat ('test46') ;    % performance test GB_mex_subassign
    logstat ('test46b') ;   % performance test GB_mex_assign
    logstat ('test47',t) ;
    logstat ('test48') ;  
    logstat ('test49') ;  
    logstat ('test50',t) ;  % test GB_mex_AxB on larger matrix
    logstat ('test51') ;    % performance test GB_mex_subassign, multiple ops
    logstat ('test51b') ;   % performance test GB_mex_assign, multiple ops
    logstat ('test52',t) ;  % performance of A*B with tall matrices, AdotB, AxB
    logstat ('test06(936)',t) ; % performance test of GrB_mxm on all semirings
    logstat ('test55',t) ;
    logstat ('test55b',t) ;
    logstat ('test56',t) ;
    logstat ('test57',t) ;
    logstat ('test58') ;  
    logstat ('test59',t) ;
    logstat ('test60',t) ;
    logstat ('test61') ;  
    logstat ('test20(1)',t) ; % long test of GB_mex_mxm on a few semirings
end

t = toc (testall_time) ;
fprintf ('\ntestall: all tests passed, total time %g sec\n', t) ;

