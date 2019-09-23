%GBCOVSTAT report status of statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global gbcov_global

if (~isempty (gbcov_global))
    covered = sum (gbcov_global > 0) ;
    not_covered = find (gbcov_global == 0) - 1
    n = length (gbcov_global) ;
    fprintf ('test coverage: %d of %d (%0.4f%%)\n', ...
        covered, n, 100 * (covered / n)) ;
end

