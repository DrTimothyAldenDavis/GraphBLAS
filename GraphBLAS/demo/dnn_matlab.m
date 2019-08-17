function Y = dnn_matlab (W, bias, Y0)
%DNN_MATLAB Sparse deep neural network in pure MATLAB
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.  Slightly revised from graphchallenge.org.

Y = Y0 ;
for i=1:length(W)
    t = tic ;

    % Propagate through layer.
    Z = Y * W {i} ;

    % Apply bias to non-zero entries.
    Y = Z + (double(logical(Z)) .* bias {i}) ;

    % Threshold negative values.
    Y (Y < 0) = 0 ;

    % Threshold maximum values.
    Y (Y > 32) = 32 ;

    t = toc (t) ;
    fprintf ('layer: %4d, nnz (Y) %8d, time %g sec\n', i, nnz (Y), t) ;
end

