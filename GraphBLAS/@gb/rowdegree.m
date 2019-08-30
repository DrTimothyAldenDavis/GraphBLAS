function D = rowdegree (X)
%GB.ROWDEGREE row degree of a GraphBLAS or MATLAB matrix.
% D(i) = # of entries in X(i,:); result is a column vector

D = gb.vreduce ('+.int64', spones (X)) ;

