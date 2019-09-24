function monoidinfo (monoid, type)
%GB.MONOIDINFO list the details of a GraphBLAS monoid.
%
% Usage
%
%   gb.monoidinfo
%   gb.monoidinfo (monoid)
%   gb.monoidinfo (monoid, type)
%
% For gb.monoidinfo(op), the op must be a string of the form
% 'op.type', where 'op' is listed below.  The second usage allows the
% type to be omitted from the first argument, as just 'op'.  This is
% valid for all GraphBLAS operations, since the type defaults to the
% type of the input matrices.  However, gb.monoidinfo does not have a
% default type and thus one must be provided, either in the op as
% gb.monoidinfo ('+.double'), or in the second argument,
% gb.monoidinfo ('+', 'double').
%
% The MATLAB interface to GraphBLAS provides for 44 different
% monoids.  The valid monoids are: '+', '*', 'max', and 'min' for all
% but the 'logical' type, and '|', '&', 'xor', and 'eq' for the
% 'logical' type.
%
% Example:
%
%   % valid monoids
%   gb.monoidinfo ('+.double') ;
%   gb.monoidinfo ('*.int32') ;
%
%   % invalid monoids
%   gb.monoidinfo ('1st.int32') ;
%   gb.monoidinfo ('abs.double') ;
%
% See also gb.unopinfo, gb.binopinfo, gb.semiringinfo,
% gb.descriptorinfo.

% FUTURE: add complex monoids

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin == 0)
    help gb.monoidinfo
elseif (nargin == 1)
    gbmonoidinfo (monoid) ;
else
    gbmonoidinfo (monoid, type) ;
end

