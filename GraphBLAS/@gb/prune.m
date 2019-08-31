function C = prune (G, identity)
%GB.PRUNE remove explicit values from a GraphBLAS matrix.
% C = gb.prune (G) removes any explicit zeros from G.
% C = gb.prune (G, identity) removes entries equal to the given identity scalar.
%
% See also gb/full.

if (nargin == 1)
    C = gb.select ('nonzero', G) ;
else
    C = gb.select ('eqthunk', G, identity) ;
end

