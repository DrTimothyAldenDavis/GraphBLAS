function G = deserialize (blob)
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
G = GrB (gbdeserialize (blob)) ;

