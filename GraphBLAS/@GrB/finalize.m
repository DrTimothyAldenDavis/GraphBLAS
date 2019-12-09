function finalize
%GRB.FINALIZE finalize SuiteSparse:GraphBLAS.
%
% Usage:
%
%   GrB.finalize
%
% GrB.finalize finishes GraphBLAS and frees all of its internal workspace.
% Use of function is optional, since all workspace is freed when MATLAB
% terminates.  No other GrB function can be called once this function
% has been called.
%
% See also: GrB.clear, GrB.init

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

gbsetup ('finish') ;

