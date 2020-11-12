function grbcov
%GRBCOV start the test coverage from scratch

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all
tstart = tic ;
system ('make purge') ;
grbmake ;
testcov ;
grbshow ;
ttotal = toc (tstart) ;

fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ttotal / 60) ;

