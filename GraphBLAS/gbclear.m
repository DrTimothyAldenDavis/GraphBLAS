function gbclear
%GBCLEAR free all internal workspace in SuiteSparse:GraphBLAS
%
% Usage:
%   gbclear
%
% This function is optional for the MATLAB interface to GraphBLAS.  Simply
% terminating the MATLAB session, or typing 'clear all' will do the same thing.
% However, if you are finished with GraphBLAS and wish to free its internal
% resources, but do not wish to free everything else freed by 'clear all', then
% use this function.  This function also clears any non-default setting of
% gbthreads.
%
% See also: clear, gbthreads

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbclear mexFunction not found; use gbmake to compile GraphBLAS') ;

