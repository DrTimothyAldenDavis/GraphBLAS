function D = coldegree (X)
%GB.COLDEGREE column degree of a GraphBLAS or MATLAB matrix.
% D = gb.coldegree (X) computes the column degrees of X.  D(j) is the
% number of entries in X(:,j).  D is returned as a column vector.
%
% See also gb/rowdegree, graph/degree, gbgraph/degree.

% FUTURE: this will be faster as a dedicated mexFunction

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

D = gb.vreduce ('+.int64', spones (X), struct ('in0', 'transpose')) ;

