function Y = dnn_gb (W, bias, Y0)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

n = size (Y0, 2) ;

% convert to GraphBLAS (this is optional, gb.* methods can take inputs
% that are either gb objects or MATLAB sparse matrices)
t = tic ;
Y0 = gb (Y0, 'single') ;
for i=1:length(W)
    W {i} = gb (W {i}, 'single') ;
    % build a gb GrB_FP32 matrix from tuples, using '+' as the dup operator
    bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
end
t = toc (t) ;
fprintf ('setup time: %g sec\n', t) ;

Y = Y0 ; % Initialize feature vectors.
for i=1:length(W) % Loop through each weight layer W{i}
    t = tic ;

    % Propagate through layer.
    Y = gb.mxm ('+.*', Y, W {i}) ;

    % Apply bias to non-zero entries:
    Y = gb.mxm ('+.+', Y, bias {i}) ;

    % Threshold negative values.
    Y = gb.select ('>0', Y) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, 32) ;
    Y = gb.assign (Y, M, 32) ;

    t = toc (t) ;
    fprintf ('layer: %4d, nnz (Y) %8d, time %g sec\n', i, nnz (Y), t) ;
end

% convert result back to MATLAB sparse matrix (also optional)
Y = sparse (Y) ;

