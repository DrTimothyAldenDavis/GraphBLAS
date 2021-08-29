function blob = serialize (G, method)
%GRB.SERIALIZE convert a matrix to a serialized blob
% blob = GrB.serialize (G) returns a uint8 array containing the contents
% of the matrix G, which may be a MATLAB or @GrB matrix.  The array may
% be saved to a binary file and used to construct a GrB_Matrix outside
% of this MATLAB/Octave interface to GraphBLAS.  It may also be used to
% reconstruct a @GrB matrix with G = GrB.deserialize (blob).
%
% blob = GrB.serialize (G,method) specifies the compression method used,
% as a string: "none": no compression, "default" or "lz4": LZ4 compression,
% ... FIXME (add more compression methods here).  The default is "lz4",
% if the method parameter is omitted.
%
% Example:
%   G = GrB (magic (5))
%   blob = GrB.serialize (G) ;      % compressed via LZ4
%   f = fopen ('G.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('G.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   G = GrB.deserialize (blob)
%
% See also GrB.deserialize, GrB.load, GrB.save, GrB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

if (isobject (G))
    % extract the contents of a GraphBLAS matrix
    G = G.opaque ;
end

% serialize the matrix into a uint8 blob
if (nargin == 1)
    % use the default compression method
    blob = gbserialize (G) ;
else
    blob = gbserialize (G, method) ;
end

