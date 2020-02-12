function clear
%GRB.CLEAR free all internal workspace in SuiteSparse:GraphBLAS.
%
% Usage:
%
%   GrB.clear
%
% GraphBLAS keeps an internal workspace to speedup its operations.  It also
% uses several global settings.  These can both be cleared with GrB.clear.
% GrB.clear also clears any non-default setting of GrB.threads, GrB.chunk, and
% GrB.format.
%
% See also: clear, GrB.init, GrB.finalize

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

gbsetup ('finish') ;
gbsetup ('start') ;

