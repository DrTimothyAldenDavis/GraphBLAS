function init
%GRB.INIT initialize SuiteSparse:GraphBLAS.
%
% Usage:
%
%   GrB.init
%
% GrB.init must be called before any GraphBLAS function is used.  The
% recommended usage of this function is to add it to your startup.m
% m-file, so it is called only once when MATLAB starts up.
%
% In case GraphBLAS is not in your path, or in case its MATLAB interface
% has not yet been compiled, then add this to your startup.m file,
% where '/home/me/GraphBLAS' is the location of your copy of GraphBLAS:
%
%       % add the MATLAB interface to the MATLAB path
%       addpath ('/home/me/GraphBLAS/GraphBLAS') :
%       try
%           GrB.init
%           fprintf ('GraphBLAS initialized\n') ;
%       catch
%           fprintf ('GraphBLAS not initialized\n') ;
%       end
%
% See also: GrB.clear, GrB.finalize, startup.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

gbsetup ('start') ;

