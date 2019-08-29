function assert (G)
%ASSERT generate an error when a condition is violated
% G must be a scalar logical (either a MATLAB or GraphBLAS scalar)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

builtin ('assert', logical (G)) ;

