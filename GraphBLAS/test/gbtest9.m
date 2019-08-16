function gbtest9
%GBTEST9 test dnn

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
levels = 4 ;
nfeatures = 6 ;
nneurons = 16 ;

for level = 1:levels
    W {level} = sprand (nneurons, nneurons, 0.5) ;
    bias {level} = -0.3 * ones (1, nneurons) ;
end

Y0 = sprandn (nfeatures, nneurons, 0.5) ;

Y1 = dnn_matlab (W, bias, Y0) ;
Y2 = dnn_gb     (W, bias, Y0) ;

err = norm (Y1-Y2,1) ;
assert (logical (err < 1e-5)) ;

fprintf ('gbtest9: all tests passed\n') ;


