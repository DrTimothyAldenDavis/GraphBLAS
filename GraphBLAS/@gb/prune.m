function C = prune (G, id)
%GB.PRUNE remove explicit values from a matrix.
% C = gb.prune (G) removes any explicit zeros from G.
% C = gb.prune (G, id) removes entries equal to the given scalar id.
%
% See also gb/full, gb.select, gb.prune.

if (nargin == 1)
    C = gb.select ('nonzero', G) ;
else
    C = gb.select ('nethunk', G, id) ;
end

