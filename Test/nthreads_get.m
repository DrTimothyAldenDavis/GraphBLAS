function nthreads = nthreads_get
%GET_NTHREADS get # of threads to use in GraphBLAS
%
% nthreads = nthreads_get

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_nthreads
if (isempty (GraphBLAS_nthreads))
    nthreads_set (1) ;
end

nthreads = GraphBLAS_nthreads ;

% fprintf ('nthreads: %d\n', GraphBLAS_nthreads) ;

