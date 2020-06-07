function C = prune (G, id)
%GRB.PRUNE remove explicit values from a matrix.
% C = GrB.prune (G) removes any explicit zeros from G.
% C = GrB.prune (G, id) removes entries equal to the given scalar id.
%
% See also GrB/full, GrB.select, GrB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO check if A is a MATLAB sparse matrix (so the prune of
% explicit zeros can be skipped, if nargin == 1 or id == 0)

if (isobject (G))
    G = G.opaque ;
end

if (nargin == 1)
    C = GrB (gbselect (G, 'nonzero')) ;
else
    if (isobject (id))
        id = id.opaque ;
    end
    C = GrB (gbselect (G, '~=', id)) ;
end

