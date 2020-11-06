function grbcov
%GRBCOV start the test coverage from scratch

system ('make purge') ;
clear all
grbmake ;
testcov ;
grbshow ;

