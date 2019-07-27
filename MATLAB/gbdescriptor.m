%GBDESCRIPTOR the GraphBLAS descriptor
%
% The GraphBLAS descriptor is a MATLAB struct that can be used to modify the
% behavior of GraphBLAS operations.  It contains the following components, each
% of which are a string or a number.  Any component of struct that is not
% present is set to the default value.  If the descriptor d is empty, or not
% present, in a GraphBLAS function, all default settings are used.
%
%   d.out   'default' or 'replace'      determines if C is cleared before
%                                       the accum/mask step
%   d.mask  'default' or 'complement'   determines if M or !M is used
%   d.in0   'default' or 'transpose'    determines A or A' is used
%   d.in1   'default' or 'transpose'    determines B or B' is used
%   d.nthreads  'default' or a number   determines # of threads to use
%
%   d.method    'default', 'Gustavson', 'heap', or 'dot'
%               determines the method used in gbmxm.
%
%   d.chunk     'default', or a number.
%               determines the number of threads to use for small problems.
%
% Refer to the GraphBLAS user guide for more details.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.
