function t = grbresults
%GRBRESULTS return time taken by last GraphBLAS function

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_results
t = GraphBLAS_results (1) ;

