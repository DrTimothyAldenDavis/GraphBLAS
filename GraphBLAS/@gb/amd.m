function p = amd (G, varargin)
%AMD approximate minimum degree ordering.
% See 'help amd' for details.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

p = builtin ('amd', logical (G), varargin {:}) ;

