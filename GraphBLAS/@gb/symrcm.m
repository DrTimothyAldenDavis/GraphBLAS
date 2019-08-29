function p = symrcm (G)
%SYMRCM Reverse Cuthill-McKee ordering.
% See 'help symrcm' for details.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

p = builtin ('symrcm', logical (G)) ;

