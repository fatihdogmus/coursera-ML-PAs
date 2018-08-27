function g = sigmoidGradient(z)
    %SIGMOIDGRADIENT returns the gradient of the sigmoid function
    %evaluated at z
    %   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    %   evaluated at z. This should work regardless if z is a matrix or a
    %   vector. In particular, if z is a vector or matrix, you should return
    %   the gradient for each element.


    % My comments: The derativation of the sigmoid function is as follows:
    % exp(-z)/(1+exp(-z)^2). This is equal to sigmoid(z) .* (1-sigmoid(z)).
    % Either can be applied, but since second method is easier to read,
    % second method is applied in here
     
    
    g = sigmoid(z) .* (1-sigmoid(z));
end
