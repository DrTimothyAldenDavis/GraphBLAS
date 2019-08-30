function D = coldegree (X)
%GB.COLDEGREE column degree of a GraphBLAS or MATLAB matrix
% D(j) = # of entries in X(:,j); result is a column vector

D = gb.vreduce ('+.int64', spones (X), struct ('in0', 'transpose')) ;

