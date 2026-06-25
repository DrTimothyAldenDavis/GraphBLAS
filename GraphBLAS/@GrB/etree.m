function [parent, varargout] = etree (G, varargin)
%ETREE elimination tree of a GraphBLAS matrix.
% See 'help etree' for details.
%
% See also GrB/amd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[parent, varargout{1:nargout-1}] = builtin ('etree', logical (G), varargin {:});

