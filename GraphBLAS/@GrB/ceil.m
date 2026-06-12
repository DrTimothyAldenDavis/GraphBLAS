function C = ceil (G)
%CEIL round entries of a matrix to nearest integers towards infinity.
%
% See also GrB/floor, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_isfloat (gbmex_type (G)) && gbmex_nvals (G) > 0) % FIXME remove nvals
    C = GrB (gbmex_apply ('ceil', G)) ;
else
    C = GrB (G) ;
end

