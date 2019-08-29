function [I, J, X] = find (G)
%FIND extract entries from a GraphBLAS matrix.
% [I, J, X] = find (G) extracts the entries from a GraphBLAS matrix G.
% X has the same type as G ('double', 'single', 'int8', ...).  I and J are
% returned as 1-based indices, the same as [I,J,X] = find (S) for a MATLAB
% matrix S.  Use gb.extracttuples to return I and J as zero-based.  Linear
% 1D indexing (I = find (S) for the MATLAB matrix S) and find (G, k, ...)
% are not supported.  G may contain explicit entries, but these are dropped
% from the output [I,J,X].  Use gb.extracttuples to return those entries.
%
% See also sparse, gb.build, gb.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

T = gbselect ('nonzero', G.opaque, struct ('kind', 'gb')) ;
if (nargout == 3)
    [I, J, X] = gbextracttuples (T) ;
elseif (nargout == 2)
    [I, J] = gbextracttuples (T) ;
else
    I = gbextracttuples (T) ;
end

