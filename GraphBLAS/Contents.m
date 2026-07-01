% @GrB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Its @GrB / @GhB interface
% provides faster sparse matrix operations than the built-in methods,
% as well as sparse integer and single-precision matrices, and
% operations with arbitrary semirings.  See 'help GrB' and 'help GhB'
% for details.
%
% The constructor methods are GrB and GhB.  If A is any matrix (GraphBLAS,
% or built-in sparse or full), then:
%
%   C = GrB (A) ;            GraphBLAS copy of a matrix A, same type
%   C = GrB (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   C = GrB (..., type) ;    create or typecast to a different type
%   C = GrB (..., format) ;  create in a specified format
%
% The type can be 'double', 'single', 'logical', 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'double complex' or
% 'single complex'.  Typical formats are 'by row' or 'by col'. 
%
% The GhB constructor is identical except that it creates a *handle* @GhB
% object, while the @GhB matrix is a MATLAB *value* object.  A handle
% object can be modified if it is an input to a MATLAB function, while a
% value object cannot.  The GhB interface can operate on matrices
% in-place, and can leave behind pending work (lazy evaluation) which
% makes it faster to use.  However, the simple MATLAB statement "C=A"
% differs; if A is a @GrB object, C is a copy of A.  If A is a @GhB handle
% object, C is another reference to the underlying object A, and modifying
% A or C will modify the other.
%
% Essentially all operators and many built-in functions are overloaded
% by the @GrB and @GhB classes, so that they can be used for GraphBLAS
% matrices.
%
% To install the GraphBLAS library and its MATLAB interface:
%
%   graphblas_install - compile SuiteSparse:GraphBLAS for MATLAB or Octave
%
% NOTE: Do not use any method in this folder with the prefix gb*.m or
% gzb*.m, or the gbmex* mexFunctions.  They are internal methods that must
% be publically visible since they are used by both @GrB and @GhB classes.
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

