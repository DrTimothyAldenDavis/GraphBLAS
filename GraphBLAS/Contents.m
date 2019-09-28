% MATLAB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Its MATLAB interface provides faster
% sparse matrix operations than the built-in methods in MATLAB, as well as
% sparse integer and single-precision matrices, and operations with arbitrary
% semirings.  See 'help GrB' for details.
%
% The constructor method is GrB.  If A is any matrix (GraphBLAS, MATLAB sparse
% or full), then:
%
%   C = GrB (A) ;            GraphBLAS copy of a matrix A, same type
%   C = GrB (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   C = GrB (..., type) ;    create or typecast to a different type
%   C = GrB (..., format) ;  create in a specified format
%
% The type can be 'double', 'single', 'logical', 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32', or 'uint64'.  The format is 'by row' or
% 'by col'.  
%
% Methods that overload the MATLAB function of the same name; at least
% one of the inputs must be a GraphBLAS matrix:
%
%   abs             fix             isreal          sign
%   all             floor           isscalar        single
%   amd             fprintf         issparse        size
%   and             full            issymmetric     sparse
%   any             graph           istril          spfun
%   assert          int16           istriu          spones
%   bandwidth       int32           isvector        sprintf
%   ceil            int64           kron            sqrt
%   colamd          int8            length          sum
%   complex         isa             logical         symamd
%   conj            isbanded        max             symrcm
%   diag            isdiag          min             tril
%   digraph         isempty         nnz             triu
%   disp            isequal         nonzeros        true
%   display         isfinite        norm            uint16
%   dmperm          isfloat         numel           uint32
%   double          ishermitian     nzmax           uint64
%   eig             isinf           ones            uint8
%   end             isinteger       prod            xor
%   eps             islogical       real            zeros
%   etree           ismatrix        repmat
%   false           isnan           reshape
%   find            isnumeric       round
%
% Operator overloading (A and/or B a GraphBLAS matrix, C a GraphBLAS matrix):
%
%     A+B    A-B   A*B    A.*B   A./B   A.\B   A.^b    A/b    C=A(I,J)
%     -A     +A    ~A     A'     A.'    A&B    A|B     b\A    C(I,J)=A
%     A~=B   A>B   A==B   A<=B   A>=B   A<B    [A,B]   [A;B]  
%     A(1:end,1:end)
%
% These built-in MATLAB methods also work with any GraphBLAS matrices:
%
%   cast isrow iscolumn ndims sprank etreeplot spy gplot
%   bicgstabl bicgstab cgs minres gmres bicg pcg qmr rjr tfqmr lsqr
%
% Static Methods: used as GrB.method; inputs can be any GraphBLAS or
% MATLAB matrix, in any combination.
%
%   apply           emult           isfull          select
%   assign          entries         issigned        semiringinfo
%   bfs             expand          ktruss          speye
%   binopinfo       extract         laplacian       subassign
%   build           extracttuples   mis             threads
%   chunk           eye             monoidinfo      tricount
%   clear           format          mxm             type
%   compact         kronecker       nonz            unopinfo
%   descriptorinfo  trans           offdiag         vreduce
%   dnn             incidence       pagerank
%   eadd            isbycol         prune
%   empty           isbyrow         reduce
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis/GraphBLAS

