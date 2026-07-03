function disp (A, level)
%DISP display the contents of a matrix.
% disp (A, level) displays the matrix A.  The 2nd argument controls how
% much is printed; 0: none, 1: terse, 2: a few entries, 3: all, 4: a few
% entries with high precision, 5: all with high precision.  The default is
% 2 if level is not present.  To use this function on a built-in matrix,
% use disp (A, GhB (level)) or GhB.print (A,level).  This is useful since
% disp(A) will always display all entries of A, which can be too verbose
% if nnz (A) is huge.
%
% Example:
%
%   A = sprand (50, 50, 0.1) ;
%   % just print a few entries
%   disp (A, GhB (2))
%   G = GhB (A)
%   % print all entries
%   A
%   disp (G, 3)
%   % print all entries in full precision
%   format long
%   A
%   disp (G, 5)
%
% See also GhB/display, GhB.print.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% the builtin disp method does not print the name of the matrix,
% while the builtin display method does.

if (nargin < 2)
    gb_display (ghb, '', A) ;
else
    gb_display (ghb, '', A, level) ;
end

