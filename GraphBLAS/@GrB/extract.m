function C = extract (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EXTRACT extract sparse submatrix.
%
% syntax for a new matrix C:                        computation:
% C = GrB.extract (A, I, J, desc)                   % C = A(I,J)
% C = GrB.extract (Cin, accum, A, I, J, desc)       % C = Cin ; C += A(I,J)
% C = GrB.extract (Cin, M, A, I, J, desc)           % C = Cin ; C<M> = A(I,J)
% C = GrB.extract (Cin, M, accum, A, I, J, desc)    % C = Cin ; C<M> += A(I,J)
%
% A is a required parameter.  All others are optional.  The arguments are
% parsed according to their type.  Arguments with different types can appear in
% any order:
%
%   Cin, M, A:  2 or 3 GraphBLAS/built-in sparse/full matrices.
%               The first three matrix inputs are Cin, M, and A.
%               If 2 matrix inputs are present, they are Cin and A.
%   accum:      an optional string
%   I,J:        cell arrays:  with no cell inputs: I = { } and J = { }.  with
%               one cell input, I is present and J = { }.  with two cell
%               inputs, I is the first cell input and J is the 2nd cell input.
%   desc:       an optional struct; must appear as the last argument
%
% desc: see 'help GrB.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%   0:  { }     This is ':', like A(:,J), refering to all m
%               rows, if A is m-by-n.
%
%   1:  { I }   1D list of row indices, like A(I,J).
%
%   2:  { start,fini }  start and fini are scalars (either double,
%               int64, or uint64).  This defines I = start:fini.
%
%   3:  { start,inc,fini } start, inc, and fini are scalars (double,
%               int64, or uint64).
%
% The J argument is identical, except that it is a list of column indices of A.
% If only one cell array is provided, J = {  } is implied, refering to all n
% columns of A, like A(I,:).  GrB.extract does not support linear indexing of a
% 2D matrix, as in C=A(I) when A is a 2D matrix.
%
% If neither I nor J are provided on input, then this implies both I = { } and
% J = { }, or A(:,:), refering to all rows and columns of A.
%
% desc.base modifies how I, start, and fini are interpretted.  If desc.base is
% 'zero-based' then they are interpretted as zero-based indices, where 0 is the
% first row or column.  If desc.base is 'one-based' (which is the default),
% then indices are intrepetted as 1-based.
%
% accum: an optional binary operator, defined by a string ('+.double') for
%   example.  This allows for C(I,J) = C(I,J) + A to be computed.  If not
%   present, no accumulator is used and C(I,J)=A is computed.  In the
%   computations listed above it is shown as "+=" but any binary operator may
%   be used.  See 'help GrB.binopinfo' for available binary operators.
%
% M: an optional mask matrix, the same size as C.
%
% Cin: an optional input matrix, containing the initial content of the
% matrix C.  If present, the C argument has size length(I)-by-length(J).
%
% All input matrices may be either
% GraphBLAS/built-in matrices, in any combination.  C is returned as a
% GraphBLAS @GrB matrix.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   I = [2 1 5]
%   J = [3 3 1 2]
%   C = GrB.extract (A, {I}, {J})
%   C2 = A (I,J)
%   C2 - C
%
% See also GrB/subsref, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 1
            [C_opaque, kind] = gbmex_extract (ghb, arg1) ;
        case 2
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_extract (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.extract (Cin, M, accum, A, I, J, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

