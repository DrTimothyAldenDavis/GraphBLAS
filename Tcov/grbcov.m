function grbcov
%GRBCOV compile, run, and evaluate test coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('grbcov, starting at %s\n', datestr (now)) ;

!rmtmph
clear all
clear mex
tstart = tic ;
system ('make purge') ;
grbmake ;
testcov ;
grbshow ;
ttotal = toc (tstart) ;

fprintf ('grbcov, ending   at %s\n', datestr (now)) ;
fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ttotal / 60) ;

