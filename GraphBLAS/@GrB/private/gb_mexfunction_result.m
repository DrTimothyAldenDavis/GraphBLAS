function C = gb_mexfunction_result (C_opaque, kind)
%GB_MEXFUNCTION_RESULT return a @GrB or MATLAB/Octave matrix from the
% C_opaque handle of a GrB_Matrix as computed by a GraphBLAS mexFunction.
% The matrix is returned as a @GrB matrix if kind is 0, or MATLAB/Octave
% otherwise.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (kind == 0)
    % return a @GrB object
    C = GrB (C_opaque) ;
else
    % return a built-in MATLAB/Octave matrix from the C_opaque handle
    C = gb2builtin (GrB (C_opaque)) ;
end

