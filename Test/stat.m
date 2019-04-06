function stat
%STAT report status of statement coverage and malloc debugging

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_debug GraphBLAS_gbcov GraphBLAS_nthreads

if (isempty (GraphBLAS_debug))
    GraphBLAS_debug = false ;
end

if (isempty (GraphBLAS_nthreads))
    GraphBLAS_nthreads = int32 (1) ;
end

fprintf ('malloc debug: %d  nthreads %d\n', ...
    GraphBLAS_debug, GraphBLAS_nthreads) ;

if (~isempty (GraphBLAS_gbcov))
    covered = sum (GraphBLAS_gbcov > 0) ;
    n = length (GraphBLAS_gbcov) ;
    fprintf ('test coverage: %d of %d (%0.4f%%)\n', ...
        covered, n, 100 * (covered / n)) ;
end

