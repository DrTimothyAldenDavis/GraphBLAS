function C = speye (varargin)
%GB.SPEYE Sparse identity matrix, of any type supported by GraphBLAS.
% C = gb.speye (...) is identical to gb.eye; see 'help gb.eye' for
% details.
%
% See also gb/eye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.eye (varargin {:}) ;

