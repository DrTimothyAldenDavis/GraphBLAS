function [have_octave, octave_version] = gb_octave
%GB_OCTAVE determine if Octave is in use, and what version.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;

if (have_octave)
    octave_version = version ;
else
    octave_version = '' ; 
end

