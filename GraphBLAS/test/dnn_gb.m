function Y = dnn_gb (W, bias, Y0)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.

n = size (Y0, 2) ;

% convert to GraphBLAS (this is optional, gb.* methods can take inputs
% that are either gb objects or MATLAB sparse matrices)
Y0 = gb (Y0, 'single') ;
for i=1:length(W)
    W {i} = gb (W {i}, 'single') ;
    % build a gb GrB_FP32 matrix from tuples, using '+' as the dup operator
    bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
end

Y = Y0 ; % Initialize feature vectors.
for i=1:length(W) % Loop through each weight layer W{i}

    % Propagate through layer.
    Y = gb.mxm ('+.*', Y, W {i}) ;

    % Apply bias to non-zero entries:
    Y = gb.mxm ('+.+', Y, bias {i}) ;

    % Threshold negative values.
    Y = gb.select ('>=0', Y) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, 32) ;
    Y = gb.assign (Y, M, 32) ;

end

% convert result back to MATLAB sparse matrix (also optional)
Y = sparse (Y) ;

