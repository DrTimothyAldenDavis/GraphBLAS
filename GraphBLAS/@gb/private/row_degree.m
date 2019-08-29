function D = row_degree (G)
% D(i) = # of entries in G(i,:); result is a column vector

D = gb.vreduce ('+.int64', spones (G)) ;

