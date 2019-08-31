function D = rowdegree (X)
%GB.ROWDEGREE row degree of a GraphBLAS or MATLAB matrix.
% D = gb.rowdegree (X) computes the row degrees of X.  D(i) is the
% number of entries in X(i,:).  D is returned as a column vector.
%
% See also gb/coldegree, graph/degree, gbgraph/degree.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

D = gb.vreduce ('+.int64', spones (X)) ;

