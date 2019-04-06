function nthreads = nthreads_set (nthreads)
%SET_NTHREADS set # of threads to use in GraphBLAS
%
% nthreads = nthreads_set (nthreads)
%
% If nthreads is empty, or if no input arguments, nthreads is set to 1.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_nthreads

if (nargin < 1)
    nthreads = [ ] ;
end

if (isempty (nthreads))
    nthreads = int32 (1) ;
end

nthreads = int32 (nthreads) ;
GraphBLAS_nthreads = nthreads ;

% fprintf ('nthreads: %d\n', GraphBLAS_nthreads) ;

