% Run the GraphBLAS demo2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% reset to the default number of threads
clear all
have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (~have_octave)
    maxNumCompThreads ('automatic') ;
end
GrB.clear ;

gbdemo2
