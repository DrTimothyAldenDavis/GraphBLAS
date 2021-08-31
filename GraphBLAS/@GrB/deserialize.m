function G = deserialize (blob, type)
%GRB.DESERIALIZE convert a serialized blob into a matrix.
% G = GrB.deserialize (blob) returns a @GrB matrix constructed from the
% uint8 array blob constructed by GrB.serialize.
%
% Example:
%   G = GrB (magic (5))
%   blob = GrB.serialize (G) ;
%   f = fopen ('G.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('G.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   G = GrB.deserialize (blob)
%
% See also GrB.serialize, GrB.load, GrB.save, GrB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% deserialize the blob into a @GrB matrix
if (nargin == 1)
    G = GrB (gbdeserialize (blob)) ;
else
    % This is feature not documented, and is only intended for testing.
    % And optional 2nd parameter allows the type of G to be specified.
    % The ANSI C rules for casting floating-point to integers is used
    % (truncation), not the MATLAB rules (rounding to nearest integer).
    G = GrB (gbdeserialize (blob, type)) ;
end

