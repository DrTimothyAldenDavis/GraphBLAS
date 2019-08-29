function C = real (G)
%REAL complex real part.
% Since all GraphBLAS matrices are currently real, real (G) is just G.
% Complex support will be added in the future.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = G ;

