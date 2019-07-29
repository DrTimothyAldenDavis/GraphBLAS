function gbsemiring (semiring, type)
%GBSEMIRING list the details of a GraphBLAS semiring, for illustration only
%
% Usage
%
%   gbsemiring (semiring)
%   gbsemiring (semiring, type)
%
% For the first usage, the semiring must be a string of the form
% 'add.mult.type', where 'add' and 'mult' are binary operators.  The second
% usage allows the type to be omitted from the first argument, as just
% 'add.mult'.  This is valid for all GraphBLAS operations, since the type
% defaults to the type of the input matrices.  However, the 
%
% The add operator must be a valid monoid, typically the operators plus, times,
% min, max, or, and, ne, xor.  The binary operator z=f(x,y) of a monoid must be
% associate and commutative, with an identity value id such that f(x,id) =
% f(id,x) = x.  Furthermore, the types of x, y, and z must all be the same.
% Thus, the '<.double' is not a valid operator for a monoid, since its output
% type (logical) does not match its inputs (double).  Thus, <.*.double is not a
% valid semiring.
%
% However, many of the binary operators are equivalent.  xor(x,y) is the same
% as minus(x,y), for example, and thus 'minus.&.logical' is the same semiring
% as as 'xor.&.logical', and both are (the same) valid semiring.
%
% Example:
%
%   % valid semirings
%   gbsemiring +.*.double
%   gbsemiring min.1st.int32
%
%   % invalid semiring (generates an error)
%   gbsemiring <.*.double
%
% gbsemiring generates an error for an invalid semiring, so user code can test
% the validity of a semiring with the MATLAB try/catch mechanism.
%
% See also gbbinops, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbsemiring mexFunction not found; use gbmake to compile GraphBLAS') ;

