function [C, I, J] = gb_compact (A, symmetric)
%GB_COMPACT: helper function for GrB.compact.
% Returns a @GrB object.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

% get the list of non-empty rows and columns
I = gb_entries (A, 'row', 'list') ;
J = gb_entries (A, 'col', 'list') ;

if (symmetric)
    I = union (I, J) ;
    J = I ;
end

% C = A (I,J)
C = GrB (gbmex_extract (ghb, A, { I }, { J })) ;

