function grbcov
%GRBCOV start the test coverage from scratch

clear all
tstart = tic ;
system ('make purge') ;
grbmake ;
testcov ;
grbshow ;
ttotal = toc (tstart) ;

fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ttotal / 60) ;

