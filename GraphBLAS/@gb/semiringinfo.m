function semiringinfo (s, type)
%GB.SEMIRINGINFO list the details of a GraphBLAS semiring.
%
% Usage
%
%   gb.semiringinfo
%   gb.semiringinfo (semiring)
%   gb.semiringinfo (semiring, type)
%
% For gb.semiring(semiring), the semiring must be a string of the form
% 'add.mult.type', where 'add' and 'mult' are binary operators.  The
% second usage allows the type to be omitted from the first argument, as
% just 'add.mult'.  This is valid for all GraphBLAS operations, since the
% type defaults to the type of the input matrices.  However,
% gb.semiringinfo does not have a default type and thus one must be
% provided, either in the semiring as gb.semiringinfo ('+.*.double'), or
% in the second argument, gb.semiringinfo ('+.*', 'double').
%
% The add operator must be a valid monoid: plus, times, min, max, and the
% boolean operators or.logical, and.logical, ne.logical, and xor.logical.
% The binary operator z=f(x,y) of a monoid must be associative and
% commutative, with an identity value id such that f(x,id) = f(id,x) = x.
% Furthermore, the types of x, y, and z for the monoid operator f must
% all be the same.  Thus, the '<.double' is not a valid monoid operator,
% since its 'logical' output type does not match its 'double' inputs, and
% since it is neither associative nor commutative.  Thus, <.*.double is
% not a valid semiring.
%
% Example:
%
%   % valid semirings
%   gb.semiringinfo ('+.*.double') ;
%   gb.semiringinfo ('min.1st.int32') ;
%
%   % invalid semiring (generates an error; since '<' is not a monoid)
%   gb.semiringinfo ('<.*.double') ;
%
% See also gb, gb.unopinfo, gb.binopinfo, gb.descriptorinfo.

% FUTURE: add complex semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin == 0)
    help gb.semiringinfo
elseif (nargin == 1)
    gbsemiringinfo (s) ;
else
    gbsemiringinfo (s, type) ;
end

