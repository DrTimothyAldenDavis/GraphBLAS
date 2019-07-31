function gbbinop (op, type)
%GBBINOP list the details of a GraphBLAS binary operator, for illustration only
%
% Usage
%
%   gbbinop (op)
%   gbbinop (op, type)
%
% For the first usage, the op must be a string of the form 'op.type', where
% 'op'.  The second usage allows the type to be omitted from the first
% argument, as just 'op'.  This is valid for all GraphBLAS operations, since
% the type defaults to the type of the input matrices.  However, this function
% does not have a default type and thus one must be provided, either in the
% op as gbbinop ('+.double'), or in the second argument, gbbinop ('+',
% 'double').
%
% Example:
%
%   % valid binary operators
%   gbbinop ('+.*.double') ;
%   gbbinop ('min.1st.int32') ;
%
%   % invalid binary operator (generates an error)
%   gbbinop ('abs.double') ;
%
% gbbinop generates an error for an invalid op, so user code can test
% the validity of an op with the MATLAB try/catch mechanism.
%
% See also gbbinops, gbnew.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbbinop mexFunction not found; use gbmake to compile GraphBLAS') ;

