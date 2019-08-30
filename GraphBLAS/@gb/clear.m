function clear
%GB.CLEAR free all internal workspace in SuiteSparse:GraphBLAS
%
% Usage:
%
%   gb.clear
%
% GraphBLAS keeps an internal workspace to speedup its operations.  It also
% uses several global settings.  These can both be cleared with gb.clear.
%
% This method is optional.  Simply terminating the MATLAB session, or
% typing 'clear all' will do the same thing.  However, if you are finished
% with GraphBLAS and wish to free its internal workspace, but do not wish
% to free everything else freed by 'clear all', then use this method.
% gb.clear also clears any non-default setting of gb.threads, gb.chunk, and
% gb.format.
%
% See also: clear, gb.threads, gb.chunk, gb.format

gbclear ;

