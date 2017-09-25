function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
s = 0;
tmp = 1;
while tmp <= m,
s = s + (theta(1,1) + theta(2,1) * X(tmp,2) - y(tmp)) ^ 2;
tmp = tmp +1;
end;

J = 1/(2*m) * s;


% =========================================================================

end
