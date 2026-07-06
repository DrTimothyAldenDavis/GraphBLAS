function [I, varargout] = find (G, varargin)
%FIND extract entries from a matrix.
% [I, J, X] = find (G) extracts the nonzeros from a matrix G.
% X has the same type as G ('double', 'single', 'int8', ...).
%
% Linear 1D indexing (I = find (S) for the built-in matrix S) is not yet
% supported.
%
% A GraphBLAS matrix G may contain explicit zero entries, and by default
% these are excluded from the result.  Use GhB.extracttuples (G) to return
% these explicit zero entries.
%
% For a column vector, I = find (G) returns I as a list of the row indices
% of nonzeros in G.  For a row vector, I = find (G) returns I as a list of
% the column indices of nonzeros in G.
%
% [...] = find (G, k, 'first') returns the first k nonozeros of G.
% [...] = find (G, k, 'last')  returns the last k nonozeros of G.
% For this usage, the first and last k are in terms of nonzeros in the
% column-major order.  Note that this usage is much slower than when 
% using a MATLAB/Octave built-in matrix, because a GraphBLAS matrix must
% have the ability to hold explicit zeros.  These must be pruned to match
% the behavior of find(G,k).
%
% The indices I and J are returned as int32 or int64 column vectors,
% depending on the dimenions of the matrix G.
%
% See also sparse, GhB.build, GhB.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[I, varargout{1:nargout-1}] = gb_find (1, G, varargin {:}) ;

