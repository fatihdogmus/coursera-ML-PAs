function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful value
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

	for iter = 1:num_iters

		J_history(iter) = computeCostMulti(X, y, theta);
		dvCost = derivationCost(m,theta,X,y);
		theta = theta - alpha*dvCost;

	end



end

function cost = derivationCost(m,theta,x,y)
    cost = 0;
    for i=1:m
        
        hypoResult = theta'*x(i,:)';
        cost = cost + ((hypoResult-y(i))*x(i,:));
    end
    cost = (cost /m)';

end
