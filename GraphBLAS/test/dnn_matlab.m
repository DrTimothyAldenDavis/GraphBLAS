function Y = dnn_matlab (W, bias, Y0)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.
Y = Y0 ; % Initialize feature vectors.
for i=1:length(W) % Loop through each weight layer W{i}
    % Propagate through layer.
    Z = Y * W {i} ;
    b = bias {i} ;
    % Apply bias to non-zero entries:
    Y = Z + (double(logical(Z)) .* b) ;
    Y (Y < 0) = 0 ; % Threshold negative values.
    Y (Y > 32) = 32 ; % Threshold maximum values.
end

