function Y = dnn_gb_terse (W, bias, Y0, option)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.

layers = length (W) ;
n = size (Y0, 2) ;

% convert to GraphBLAS (this is optional, gb.* methods can take inputs
% that are either gb objects or MATLAB sparse matrices)
if (nargin > 3 && isequal (option, 'gb'))
    fprintf ('convert to gb single\n') ;
    n = size (Y0, 2) ;
    Y0 = gb (Y0, 'single') ;
    for i=1:length(W)
        W {i} = gb (W {i}, 'single') ;
        % build a gb GrB_FP32 matrix from tuples, using '+' as the dup operator
        bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
    end
else
    % this works too, except that bias{i} is a row vector in the MATLAB
    % version of the problem; convert it to a diagonal matrix
    fprintf ('do not convert to gb single; leave as MATLAB double sparse\n') ;
    for i = 1:layers
        bias {i} = spdiags (bias {i}', 0, n, n) ;
    end
end

W1 = W {1} ;
bias1 = bias {1} ;
whos

Y = Y0 ; % Initialize feature vectors.
for i=1:layers

    % Propagate through layer, Apply bias to non-zero entries, and Threshold
    % negative values.
    Y = gb.select ('>=0', gb.mxm ('+.+', gb.mxm ('+.*', Y, W {i}), bias {i})) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, 32) ;
    if (nnz (M) > 0)
        Y = gb.assign (Y, M, 32) ;
    end

end

% convert result back to MATLAB sparse matrix (also optional)
Y = sparse (Y) ;

