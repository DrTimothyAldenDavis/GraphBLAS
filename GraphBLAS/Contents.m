% MATLAB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Its MATLAB interface provides faster
% sparse matrix operations than the built-in methods in MATLAB, as well as
% sparse integer and single-precision matrices, and operations with arbitrary
% semirings.  See 'help gb' for details.
%
% The constructor method is gb.  If A is any matrix (GraphBLAS, MATLAB sparse
% or full), then:
%
%   G = gb (A) ;            GraphBLAS copy of a matrix A, same type
%   G = gb (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   G = gb (..., type) ;    create or typecast to a different type
%   G = gb (..., format) ;  create in a specified format
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
%   amd             full            issparse        size
%   and                             issymmetric     sparse
%   any             graph           istril          spfun
%   assert          int16           istriu          spones
%   bandwidth       int32           isvector        sqrt
%   ceil            int64           kron            sum
%   colamd          int8            length          symamd
%   complex         isa             logical         symrcm
%   conj            isbanded        max             tril
%   diag            isdiag          min             triu
%   digraph         isempty         nnz             true
%   disp            isequal         nonzeros        uint16
%   display         isfinite        norm            uint32
%   dmperm          isfloat         numel           uint64
%   double          ishermitian     nzmax           uint8
%   eig             isinf           ones            xor
%   end             isinteger       prod            zeros
%   eps             islogical       real
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
% Static Methods: used as gb.method; inputs can be any GraphBLAS or
% MATLAB matrix.
%
%   apply           emult           issigned        select
%   assign          expand          ktruss          semiringinfo
%   bfs             extract         laplacian       speye
%   binopinfo       extracttuples   mis             subassign
%   build           eye             monoidinfo      threads
%   chunk           format          mxm             tricount
%   clear           gbkron          nvals           type
%   coldegree       gbtranspose     offdiag         unopinfo
%   descriptorinfo  incidence       pagerank        vreduce
%   dnn             isbycol         prune
%   eadd            isbyrow         reduce
%   empty           isfull          rowdegree
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis/GraphBLAS

