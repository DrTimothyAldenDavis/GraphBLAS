function [V, varargout] = eig (G, varargin)
%EIG Eigenvalues and eigenvectors of a GraphBLAS matrix.
% See 'help eig' for details.
%
% See also eigs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% convert G to a built-in matrix
if (isreal (G) && issymmetric (G))
    % G can be sparse if G is real and symmetric
    A = double (G) ;
else
    % otherwise, G must be full.
    A = full (double (G)) ;
end

% use the built-in eig
if (nargin == 1)
    [V, varargout{1:nargout-1}] = builtin ('eig', A) ;
else
    args = varargin ;
    for k = 1:length (args)
        argk = args {k} ;
        if (isobject (argk))
            args {k} = full (double (argk)) ;
        end
    end
    [V, varargout{1:nargout-1}] = builtin ('eig', A, args {:}) ;
end

