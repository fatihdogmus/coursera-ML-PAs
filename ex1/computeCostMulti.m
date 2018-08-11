function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
	m = length(y); % number of training examples

% You need to return the following variables correctly 
	J = 0;

	    sumResult = 0;
    for i=1:m
        hypoResult = calculateHypothesis(X(i,:),theta);
        sumResult = sumResult + (hypoResult - y(i))^2;
    end
    J = sumResult/(2*m);

end

function hypothesesiResult = calculateHypothesis(input,theta)
    hypothesesiResult = theta'*input';


end
