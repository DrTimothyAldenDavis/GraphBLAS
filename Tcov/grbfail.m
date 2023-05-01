function grbfail
%GRBFAIL compile, run, and evaluate test coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

!rmtmph
clear all
clear mex
tstart = tic ;
system ('make purge') ;
grbmake ;

try
    addpath ('../Test') ;
    addpath ('../Test/spok') ;
    cd ../Test/spok
    spok_install ;
    cd ../../Tcov
    mex -g -R2018a ../Test/GB_spones_mex.c
    debug_on ;
    grbcover ;
    testfail ;
catch me
    debug_off ;
    rethrow (me) ;
end

% grbshow ;
ttotal = toc (tstart) ;

fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ttotal / 60) ;

