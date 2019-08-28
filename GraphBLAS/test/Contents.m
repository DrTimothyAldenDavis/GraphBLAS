% SuiteSparse/GraphBLAS/GraphBLAS/test: tests for GraphBLAS MATLAB interface
%
% To run all the tests, just use gbtestall.
%
%  gbtestall - test GraphBLAS MATLAB interface
%
%  gbtest1   - test gb
%  gbtest2   - list all binary operators
%  gbtest3   - test dnn
%  gbtest4   - list all 1865 possible semirings
%  gbtest5   - test gb.descriptorinfo
%  gbtest6   - test gb.mxm
%  gbtest7   - test gb.build
%  gbtest8   - test gb.select
%  gbtest9   - test eye and speye
%  gbtest10  - test gb.assign
%  gbtest11  - test gb, sparse
%  gbtest12  - test gb.eadd, gb.emult
%  gbtest13  - test find and gb.extracttuples
%  gbtest14  - test gb.gbkron
%  gbtest15  - list all unary operators
%  gbtest16  - test gb.extract
%  gbtest17  - test gb.gbtranspose
%  gbtest18  - test comparators
%  gbtest19  - test mpower
%  gbtest20  - test bandwidth, isdiag, ceil, floor, round, fix
%  gbtest21  - test isfinite, isinf, isnan
%  gbtest22  - test reduce to scalar
%  gbtest23  - test min and max
%  gbtest24  - test any, all
%  gbtest25  - test diag, tril, triu
%  gbtest26  - test typecasting
%  gbtest27  - test conversion to full
%  gbtest28  - test gb.build
%  gbtest29  - test subsref and subsasgn with logical indexing
%  gbtest30  - test colon notation
%  gbtest31  - test gb and casting
%  gbtest32  - test nonzeros
%  gbtest33  - test spones, numel, nzmax, size, length, isempty, issparse, ...
%  gbtest34  - test repmat
%  gbtest35  - test reshape
%  gbtest36  - test abs, sign
%  gbtest37  - test istril, istriu, isbanded, isdiag, ishermitian, ...
%  gbtest38  - test sqrt, eps, ceil, floor, round, fix, real, conj, ...
%  gbtest39  - test amd, colamd, symamd, symrcm, dmperm, etree
%  gbtest40  - test sum, prod, max, min, any, all, norm
%
% Utility functions:
%
%  gbtest_binops - return a cell array of strings, listing all binary operators
%  gbtest_types  - return a cell array of strings, listing all types

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

