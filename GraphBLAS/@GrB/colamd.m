function [p, varargout] = colamd (G, varargin)
%COLAMD column approximate minimum degree ordering.
% See 'help colamd' for details.
%
% See also GrB/amd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[p, varargout{1:nargout-1}] = colamd (double (G), varargin {:}) ;

