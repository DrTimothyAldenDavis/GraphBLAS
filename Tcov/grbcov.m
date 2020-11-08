function grbcov
%GRBCOV start the test coverage from scratch

tstart = tic ;
system ('make purge') ;
clear all
grbmake ;
testcov ;
grbshow ;
ttotal = tic (tstart) ;

fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ...
    ttotal / 60) ;

