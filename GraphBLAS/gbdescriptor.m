function gbdescriptor (d)
%GBDESCRIPTOR list the contents of a SuiteSparse:GraphBLAS descriptor
%
% Usage:
%
%   gbdescriptor
%   gbdescriptor (d)
%
% The GraphBLAS descriptor is a MATLAB struct that can be used to modify the
% behavior of GraphBLAS operations.  It contains the following components, each
% of which are a string or a number.  Any component of struct that is not
% present is set to the default value.  If the descriptor d is empty, or not
% present, in a GraphBLAS function, all default settings are used.
%
% The following descriptor values are strings:
%
%   d.out     'default' or 'replace'      determines if C is cleared before
%                                         the accum/mask step
%   d.mask    'default' or 'complement'   determines if M or !M is used
%   d.in0     'default' or 'transpose'    determines A or A' is used
%   d.in1     'default' or 'transpose'    determines B or B' is used
%   d.axb     'default', 'Gustavson', 'heap', or 'dot'
%              determines the method used in gbmxm.  The default is to let
%              GraphBLAS determine the method automatically, via a heuristic.
%
% These are scalar values:
%
%   d.nthreads  max # of threads to use.  The default is omp_get_max_threads.
%   d.chunk     controls # of threads to use for small problems.
%
% This function simply lists the contents of a GraphBLAS descriptor and checks
% if its contents are valid.
%
% Refer to the SuiteSparse:GraphBLAS User Guide for more details.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbdescriptor mexFunction not found; use gbmake to compile GraphBLAS') ;

