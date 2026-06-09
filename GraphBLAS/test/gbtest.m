function gbtest
%GBTEST test GraphBLAS interface
% First compile the GraphBLAS library by typing 'make' in the top-level
% GraphBLAS folder, in your system shell.  That statement will use cmake to
% compile GraphBLAS.  Use 'make JOBS=40' to compile in parallel (replace '40'
% with the number of cores in your system).  Next, do the following:
%
% This test has been ported to Octave 7, as of SuiteSparse:GraphBLAS v5.1.  A
% few features differ between Octave and MATLAB, so those tests are skipped for
% Octave.  Octave passes all of the essential tests below.
%
% Example:
%
%   cd GraphBLAS/GraphBLAS
%   addpath (pwd) ;
%   savepath ;          % if this fails, edit your startup.m file
%   cd @GrB/private
%   gbmake ;            % compile the interface to GraphBLAS
%   cd ../../test
%   gbtest              % run this test
%
% See also GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% gbtest3 requires ../demo/dnn_builtin.m and ../demo/dnn_builtin2gb.m.
demo_folder = fullfile (fileparts (mfilename ('fullpath')), '../demo') ;
addpath (demo_folder) ;
rng ('default') ;

have_octave = gb_octave ;

gbtest0   % test GrB.clear
assert (GrB.nmalloc == 0) ;
gbtest1   % test GrB
assert (GrB.nmalloc == 0) ;
gbtest2   % list all binary operators
assert (GrB.nmalloc == 0) ;
gbtest3   % test dnn
assert (GrB.nmalloc == 0) ;
gbtest4   % list all possible semirings
assert (GrB.nmalloc == 0) ;
gbtest5   % test GrB.descriptorinfo
assert (GrB.nmalloc == 0) ;
gbtest6   % test GrB.mxm
assert (GrB.nmalloc == 0) ;
gbtest7   % test GrB.build
assert (GrB.nmalloc == 0) ;
gbtest8   % test GrB.select
assert (GrB.nmalloc == 0) ;
gbtest9   % test eye and speye
assert (GrB.nmalloc == 0) ;
gbtest10  % test GrB.assign
assert (GrB.nmalloc == 0) ;
gbtest11  % test GrB, sparse
assert (GrB.nmalloc == 0) ;
gbtest12  % test GrB.eadd, GrB.emult, GrB.eunion
assert (GrB.nmalloc == 0) ;
gbtest13  % test find and GrB.extracttuples
assert (GrB.nmalloc == 0) ;
gbtest14  % test kron and GrB.kronecker
assert (GrB.nmalloc == 0) ;
gbtest15  % list all unary operators
assert (GrB.nmalloc == 0) ;
gbtest16  % test GrB.extract
assert (GrB.nmalloc == 0) ;
gbtest17  % test GrB.trans
assert (GrB.nmalloc == 0) ;
gbtest18  % test comparators (and, or, >, ...)
assert (GrB.nmalloc == 0) ;
gbtest19  % test mpower
assert (GrB.nmalloc == 0) ;
gbtest20  % test bandwidth, isdiag, ceil, floor, round, fix
assert (GrB.nmalloc == 0) ;
gbtest21  % test isfinite, isinf, isnan
assert (GrB.nmalloc == 0) ;
gbtest22  % test reduce to scalar
assert (GrB.nmalloc == 0) ;
gbtest23  % test min and max
assert (GrB.nmalloc == 0) ;
gbtest24  % test any, all
assert (GrB.nmalloc == 0) ;
gbtest25  % test diag, tril, triu
assert (GrB.nmalloc == 0) ;
gbtest26  % test typecasting
assert (GrB.nmalloc == 0) ;
gbtest27  % test conversion to full
assert (GrB.nmalloc == 0) ;
gbtest28  % test GrB.build
assert (GrB.nmalloc == 0) ;
gbtest29  % test subsref and subsasgn with logical indexing
assert (GrB.nmalloc == 0) ;
gbtest30  % test colon notation
assert (GrB.nmalloc == 0) ;
gbtest31  % test GrB and casting
assert (GrB.nmalloc == 0) ;
gbtest32  % test nonzeros
assert (GrB.nmalloc == 0) ;
gbtest33  % test spones, numel, nzmax, size, length, isempty, issparse, ...
assert (GrB.nmalloc == 0) ;
gbtest34  % test repmat
assert (GrB.nmalloc == 0) ;
gbtest35  % test reshape
assert (GrB.nmalloc == 0) ;
gbtest36  % test abs, sign
assert (GrB.nmalloc == 0) ;
gbtest37  % test istril, istriu, isbanded, isdiag, ishermitian, ...
assert (GrB.nmalloc == 0) ;
gbtest38  % test sqrt, eps, ceil, floor, round, fix, real, conj, ...
assert (GrB.nmalloc == 0) ;
gbtest39  % test amd, colamd, symamd, symrcm, dmperm, etree
assert (GrB.nmalloc == 0) ;
gbtest40  % test sum, prod, max, min, any, all, norm
assert (GrB.nmalloc == 0) ;
gbtest41  % test ones, zeros, false
assert (GrB.nmalloc == 0) ;
gbtest42  % test for nan
assert (GrB.nmalloc == 0) ;
gbtest43  % test error handling
assert (GrB.nmalloc == 0) ;
gbtest44  % test subsasgn, mtimes, plus, false, ...
assert (GrB.nmalloc == 0) ;
gbtest45  % test GrB.vreduce
assert (GrB.nmalloc == 0) ;
gbtest46  % test GrB.subassign and GrB.assign
assert (GrB.nmalloc == 0) ;
gbtest47  % test GrB.entries, GrB.nonz, numel
assert (GrB.nmalloc == 0) ;
gbtest48  % test GrB.apply
assert (GrB.nmalloc == 0) ;
gbtest49  % test GrB.prune
assert (GrB.nmalloc == 0) ;
gbtest50  % test GrB.ktruss and GrB.tricount
assert (GrB.nmalloc == 0) ;
gbtest51  % test GrB.tricount
assert (GrB.nmalloc == 0) ;
gbtest52  % test GrB.format
assert (GrB.nmalloc == 0) ;
gbtest53  % test GrB.monoidinfo
assert (GrB.nmalloc == 0) ;
gbtest54  % test GrB.compact
assert (GrB.nmalloc == 0) ;
gbtest55  % test disp
assert (GrB.nmalloc == 0) ;
gbtest56  % test GrB.empty
assert (GrB.nmalloc == 0) ;
gbtest57  % test fprintf and sprintf
assert (GrB.nmalloc == 0) ;
gbtest58  % test uplus
assert (GrB.nmalloc == 0) ;
gbtest59  % test end
assert (GrB.nmalloc == 0) ;
gbtest60  % test issigned
assert (GrB.nmalloc == 0) ;
gbtest62  % test ldivide, rdivide, mldivide, mrdivide
assert (GrB.nmalloc == 0) ;
gbtest65  % test GrB.mis
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest61  % test GrB.laplacian
    assert (GrB.nmalloc == 0) ;
    gbtest63  % test GrB.incidence
    assert (GrB.nmalloc == 0) ;
    gbtest64  % test GrB.pagerank
    assert (GrB.nmalloc == 0) ;
    gbtest66  % test graph
    assert (GrB.nmalloc == 0) ;
    gbtest67  % test digraph
    assert (GrB.nmalloc == 0) ;
end

gbtest68  % test isequal
assert (GrB.nmalloc == 0) ;
gbtest69  % test flip
assert (GrB.nmalloc == 0) ;
gbtest70  % test GrB.random
assert (GrB.nmalloc == 0) ;
gbtest71  % test GrB.selectopinfo
assert (GrB.nmalloc == 0) ;
gbtest72  % test any-pair semiring
assert (GrB.nmalloc == 0) ;
gbtest73  % test GrB.normdiff
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave returns double, MATLAB returns integer.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest74  % test bitwise operators
    assert (GrB.nmalloc == 0) ;
    gbtest75  % test bitshift
    assert (GrB.nmalloc == 0) ;
end

gbtest76  % test trig functions
assert (GrB.nmalloc == 0) ;
gbtest77  % test error handling
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave: bit index must be in proper range.
    % MATLAB: bit indices outside the size of the integer are ignored.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest78  % test integer operations
    assert (GrB.nmalloc == 0) ;
end

gbtest79  % test power
assert (GrB.nmalloc == 0) ;
gbtest80  % test complex division and power
assert (GrB.nmalloc == 0) ;
gbtest81  % test complex operators
assert (GrB.nmalloc == 0) ;
gbtest82  % test complex A*B, A'*B, A*B', A'*B', A+B
assert (GrB.nmalloc == 0) ;
gbtest83  % test GrB.apply
assert (GrB.nmalloc == 0) ;
gbtest84  % test GrB.assign
assert (GrB.nmalloc == 0) ;
gbtest85  % test GrB.subassign
assert (GrB.nmalloc == 0) ;
gbtest86  % test GrB.mxm
assert (GrB.nmalloc == 0) ;
gbtest87  % test GrB.eadd
assert (GrB.nmalloc == 0) ;
gbtest88  % test GrB.emult
assert (GrB.nmalloc == 0) ;
gbtest89  % test GrB.extract
assert (GrB.nmalloc == 0) ;
gbtest90  % test GrB.reduce
assert (GrB.nmalloc == 0) ;
gbtest91  % test GrB.trans
assert (GrB.nmalloc == 0) ;
gbtest92  % test GrB.kronecker
assert (GrB.nmalloc == 0) ;
gbtest93  % test GrB.select
assert (GrB.nmalloc == 0) ;
gbtest94  % test GrB.vreduce
assert (GrB.nmalloc == 0) ;
gbtest95  % test indexing
assert (GrB.nmalloc == 0) ;
gbtest97  % test GrB.apply2
assert (GrB.nmalloc == 0) ;
gbtest98  % test row/col degree for hypersparse matrices
assert (GrB.nmalloc == 0) ;
gbtest99  % test performance of C=A'*B and C=A'
assert (GrB.nmalloc == 0) ;
gbtest100 % test GrB.ver and GrB.version
assert (GrB.nmalloc == 0) ;
if (~have_octave)
    % octave cannot load the mat file from MATLAB with a v3 @GrB object
    gbtest101 % test loading of v3 GraphBLAS objects
    assert (GrB.nmalloc == 0) ;
end
gbtest102 % test horzcat, vertcat, cat, cell2mat
assert (GrB.nmalloc == 0) ;
gbtest103 % test iso matrices
assert (GrB.nmalloc == 0) ;
gbtest104 % test formats
assert (GrB.nmalloc == 0) ;
gbtest105 % test logical assignment with iso matrices
assert (GrB.nmalloc == 0) ;
gbtest106 % test build
assert (GrB.nmalloc == 0) ;
gbtest107 % test cell2mat error handling
assert (GrB.nmalloc == 0) ;
gbtest108 % test mat2cell
assert (GrB.nmalloc == 0) ;
gbtest109 % test num2cell
assert (GrB.nmalloc == 0) ;
gbtest110 % test argmax
assert (GrB.nmalloc == 0) ;
gbtest111 % test argmin
assert (GrB.nmalloc == 0) ;
gbtest112 % test load and save
assert (GrB.nmalloc == 0) ;
gbtest113 % test ones and eq
assert (GrB.nmalloc == 0) ;
gbtest114 % test kron with iso matrices
assert (GrB.nmalloc == 0) ;
gbtest115 % test serialize/deserialize
assert (GrB.nmalloc == 0) ;
gbtest116 % test GrB.binopinfo for index_unary operators
assert (GrB.nmalloc == 0) ;
gbtest117 % test idxunop in GrB.apply2
assert (GrB.nmalloc == 0) ;
gbtest118 % test GrB.argsort
assert (GrB.nmalloc == 0) ;
gbtest119 % test GrB.eunion
assert (GrB.nmalloc == 0) ;
gbtest120 % test subsref
assert (GrB.nmalloc == 0) ;
gbtest121 % test times with scalars
assert (GrB.nmalloc == 0) ;
gbtest122 % test reshape
assert (GrB.nmalloc == 0) ;
gbtest123 % test reshape
assert (GrB.nmalloc == 0) ;
gbtest124 % test binops
assert (GrB.nmalloc == 0) ;
gbtest125 % test monoids
assert (GrB.nmalloc == 0) ;
gbtest126 % test selectops
assert (GrB.nmalloc == 0) ;
gbtest127 % test semirings
assert (GrB.nmalloc == 0) ;
gbtest128 % test unops
assert (GrB.nmalloc == 0) ;
gbtest129 % test jit
assert (GrB.nmalloc == 0) ;
gbtest130 % test argmin and argmax
assert (GrB.nmalloc == 0) ;
gbtest131 % misc error handling
assert (GrB.nmalloc == 0) ;
gbtest132 % test load/save from prior versions of GraphBLAS
assert (GrB.nmalloc == 0) ;
gbtest96  % test GrB.optype
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest00  % test GrB.bfs and plot (graph (G))
    assert (GrB.nmalloc == 0) ;
end

% restore default # of threads
demo_nproc ;
assert (GrB.nmalloc == 0) ;
GrB.clear
assert (GrB.nmalloc == 0) ;

fprintf ('\ngbtest: all tests passed\n') ;

if (GrB.nmalloc > 0)
    error ('memory leak!  %d\n', GrB.nmalloc) ;
end

