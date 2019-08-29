function result = isdiag (G)
%ISDIAG True if G is a diagonal matrix.

% FUTURE: this will be much faster when 'bandwidth' is a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

result = isbanded (G, 0, 0) ;

