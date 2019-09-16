function Cout = subassign (varargin)
%GB.SUBASSIGN: assign a submatrix into a matrix.
%
% gb.subassign is an interface to GxB_Matrix_subassign and
% GxB_Matrix_subassign_[TYPE], computing the GraphBLAS expression:
%
%   C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')
%
% where A can be a matrix or a scalar.
%
% Usage:
%
%   Cout = gb.subassign (Cin, M, accum, A, I, J, desc)
%
%   Cin and A are required parameters.  All others are optional.
%   The arguments are parsed according to their type.  Arguments
%   with different types can appear in any order.
%       Cin, M, A:  2 or 3 GraphBLAS or MATLAB sparse/full matrices.
%                   The first three matrix inputs are Cin, M, and A.
%                   If 2 matrix inputs are present, they are Cin and A.
%       accum:      an optional string
%       I,J:        cell arrays:  with no cell inputs: I = { } and J = { }.
%                   with one cell input, I is present and J = { }.
%                   with two cell inputs, I is the first cell input and J
%                   is the second cell input.
%       desc:       an optional struct.
%
% gb.subassign is identical to gb.assign, with two key differences:
%
%   (1) The mask is different.
%       With gb.subassign, the mask M is length(I)-by-length(J),
%       and M(i,j) controls how A(i,j) is assigned into C(I(i),J(j)).
%       With gb.assign, the mask M has the same size as C,
%       and M(i,j) controls how C(i,j) is assigned.
%   (2) The d.out = 'replace' option differs.  gb.assign can clear
%       entries outside the C(I,J) submatrix; gb.subassign cannot.
%
% If there is no mask, or if I and J are ':', then the two methods are
% identical.  The examples shown in 'help gb.assign' also work with
% gb.subassign.  Otherwise, gb.subassign is faster.  The two methods are
% described below, where '+' is the optional accum operator.
%
%   step  | gb.assign       gb.subassign
%   ----  | ---------       ------------
%   1     | S = C(I,J)      S = C(I,J)
%   2     | S = S + A       S<M> = S + A
%   3     | Z = C           C(I,J) = S
%   4     | Z(I,J) = S
%   5     | C<M> = Z
%
% Refer to gb.assign for a description of the other input/outputs.
%
% See also gb.assign, subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbsubassign (args {:})) ;
else
    Cout = gbsubassign (args {:}) ;
end

