function [I, whole] = gb_index (I)
%GB_INDEX helper function for subsref and subsasgn
%
% [I, whole] = gb_index (I) converts I into a cell array of MATLAB
% matrices or vectors containing integer indices:
%
%   I = { }: this denotes A(:), accessing all rows or all columns.
%       In this case, the parameter whole is returned as true.
%
%   I = { list }: denotes A(list)
%
%   I = { start,fini }: denotes A(start:fini), without forming
%       the explicit list start:fini.
%
%   I = { start,inc,fini }: denotes A(start:inc:fini), without forming
%       the explicit list start:inc:fini.
%
% The input I can be a GraphBLAS matrix (as an object or its opaque
% struct).  In this case, it is wrapped in a cell, I = {subsindex(I)},
% but kept as 1-based indices (they are later translated to 0-based).
%
% If the input is already a cell array, then it is already in one of the
% above forms.  Any member of the cell array that is a GraphBLAS matrix or
% struct is converted into an index list, with subsindex(I{k}).
%
% MATLAB passes the string I = ':' to the subsref and subsasgn methods.
% This is converted into I = { }.
%
% If I is a MATLAB matrix or vector (not a cell array), then it is
% wrapped in a cell array, { I }, to denote A(I).
%
% See also GrB/subsindex, GrB/subsref, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

whole = false ;

if (isobject (I))

    % C (I) where I is a GraphBLAS matrix or vector containing integer
    % indices (as an opaque object).  Do not yet subtract 1 from the
    % indices; this will be done internally in gbextract.
    I = I.opaque ;
    I = { (gb_subsindex (I, 0)) } ;

elseif (isstruct (I))

    % C (I) where I is a GraphBLAS struct.  Do not yet subtract
    % 1 from the indices; this will be done internally in gbextract.
    I = { (gb_subsindex (I, 0)) } ;

elseif (iscell (I))

    % The index I already appears as a cell, for the usage
    % C ({ }), C ({ I }), C ({start,fini}), or C ({start,inc,fini}).
    len = length (I) ;
    if (len > 3)
        error ('invalid indexing: usage is A ({start,inc,fini})') ;
    elseif (len == 0)
        % C ({ })
        whole = true ;
    else
        % C ({ I }), C ({start,fini}), or C ({start,inc,fini})
        for k = 1:length(I)
            K = I {k} ;
            if (isobject (K))
                % C ({ ..., K, ... }) where K is a GraphBLAS object
                K = K.opaque ;
                I {k} = gb_subsindex (K, 0) ;
            elseif (isstruct (K))
                % C ({ ..., K, ... }) where I is a GraphBLAS struct
                I {k} = gb_subsindex (K, 0) ;
            end
        end
    end

elseif (ischar (I) && isequal (I, ':'))

    % C (:)
    I = { } ;
    whole = true ;

else

    % C (I) where I is a MATLAB matrix or vector containing integer
    % indices
    I = { I } ;

end

