function disp (H, level)
%DISPLAY display the contents of a gbgraph.
% display (H) displays the attributes and first few entries of a gbgraph.
% Use disp(H,3) to display all of the content of H.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    level = 2 ;
end

gbgraph_display (H, inputname (1), level) ;

