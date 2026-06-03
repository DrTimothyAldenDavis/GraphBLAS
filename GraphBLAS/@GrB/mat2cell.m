function C = mat2cell (A, m, n)
%MAT2CELL Break matrix up into a cell array of matrices.
%
% C = mat2cell (A,m,n) breaks up the 2D @GrB matrix A into a cell array of
% submatrices, where C{i,j} has size m(i)-by-n(j), and where sum(m) must
% equal the number of rows of A, and sum(n) must equal the number of columns
% of A.  The size of C is length(m)-by-length(n).
%
% Example:
%   A = GrB ([1 2 3 4; 5 6 7 8; 9 10 11 12]) ;
%   C = mat2cell (A, [1 2], [1 3])
%
% The example creates a 2-by-2 cell array C, where:
%
%   C{1,1} = A (1,1) ;      % of size 1-by-1
%   C{1,2} = A (1,2:3) ;    % of size 1-by-3
%   C{2,1} = A (2:3,1) ;    % of size 2-by-1
%   C{2,2} = A (2:3,2:3) ;  % of size 2-by-3
%
% If the 3rd argument n is not present, it defaults to size(A,2), so that
% C is a length(m)-by-1 cell array of matrices, and C{i} has size
% m(i)-by-size(A,2).
%
% See also GrB/horzcat, GrB/vertcat, GrB/cat, GrB.cell2mat, GrB/num2cell.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    if (nargin < 3)
        [~, n] = gbsize (A) ;
    end
else
    m = double (m) ;
    if (nargin < 3)
        n = size (A, 2) ;
    end
end

% S is returned as a cell array of @GrB opaque handle structs.
S = gbsplit (A, m, n) ;

% convert each entry in S to a @GrB object
C = cell (size (S)) ;
for k = 1:numel(C)
    C {k} = GrB (S {k}) ;
end

% NOTE: this method has a near zero chance of causing a memory leak.  If one of
% the conversions C {k} = GrB (S {k}) fails, this method returns immediately.
% MATLAB will know how to delete all @GrB objects in C, by calling gbdelete.
% It will not know how to properly delete all of the @GrB handle structs in S
% that remain.  These point to GraphBLAS matrices in malloc/free space, so this
% will cause a leak.  However, this failure is remote.  Each conversion of
% GrB (S {k}) allocates a very small amount of memory and is unlikely to fail.

