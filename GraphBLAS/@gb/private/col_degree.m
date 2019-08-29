function D = col_degree (G)
% D(j) = # of entries in G(:,j); result is a column vector

D = gb.vreduce ('+.int64', spones (G), struct ('in0', 'transpose')) ;

