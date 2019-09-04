function C = prune (G, s)
%GB.PRUNE remove explicit values from a GraphBLAS matrix.
% C = gb.prune (G) removes any explicit zeros from G.
% C = gb.prune (G, s) removes entries equal to the given scalar s.
%
% See also gb/full.

if (nargin == 1)
    C = gb.select ('nonzero', G) ;
else
    C = gb.select ('nethunk', G, s) ;
end

