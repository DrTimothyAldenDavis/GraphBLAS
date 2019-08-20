function gbtestall
%GBTESTALL test GraphBLAS MATLAB interface

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% gbtest9 requires ../demo/dnn_gb.m and dnn_matlab.m
addpath ../demo

gbtest1
gbtest2
gbtest3
gbtest4
gbtest5
gbtest6
gbtest7
gbtest8
gbtest9
gbtest10
gbtest11
gbtest12
gbtest13
gbtest14
gbtest15
gbtest16
gbtest17
gbtest18
gbtest19
gbtest20
gbtest21
gbtest22
gbtest23
gbtest24
gbtest25
gbtest26
gbtest27
gbtest28
gbtest29
gbtest30

fprintf ('\ngbtestall: all tests passed\n') ;

