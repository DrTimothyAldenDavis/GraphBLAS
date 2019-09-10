function Y = dnn (W, bias, Y0)
%GB.DNN Sparse deep neural network in GraphBLAS.
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W, and
% bias vectors.  The input features are in a matrix Y0 of size nfeatures-by-
% nneurons.  The DNN weights W is a cell array with W{k} being the kth layer of
% the DNN, so that the number of layers is nlayers = length (W).  W{k} is a
% matrix of size nneurons-by-nneurons.  The bais variable is a cell array of
% length nlayers.  Each bias{k} is a diagonal matrix of size nneurons-by-
% nneurons, which gives the bias values of each neuron in the kth layer.
%
% Usage:
%
%   Y = gb.dnn (W, bias, Y0) ;
%
% The matrices can be stored by row or by column, but gb.format ('by row')
% is significantly faster.  For the 2019 GraphChallenge, all matrices can be
% 'single', and the same results are obtained.
%
% In the MATLAB reference implementation, the bias{k} is a row vector of size
% 1-by-nneurons.  The MATLAB reference inputs can be converted to GraphBLAS
% matrices with the following code:
%
%   d = struct ('format', 'by row') ;
%   n = size (Y0, 2) ;
%   Y0 = gb (Y0, 'single', 'by row') ;
%   for k=1:length(W)
%       W {k} = gb (W {k}, 'single', 'by row') ;
%       bias {k} = gb.build (1:n, 1:n, bias {k}, n, n, '+', 'single', d) ;
%   end
%
% All of the above conversion is optional, except for bias {k} since it is
% changed from a row vector to a diagonal matrix.
%
% See also dnn_matlab, dnn_mat2gb in GraphBLAS/demo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

Y = Y0 ;
for k = 1:length(W)

    % Propagate through layer, apply bias, and threshold negative values.
    Y = gb.select ('>0', gb.mxm ('+.+', Y * W {k}, bias {k})) ;
     
    M = Y > 32 ;
    if (nnz (M) > 0)
        Y (M) = 32 ;
    end

end

