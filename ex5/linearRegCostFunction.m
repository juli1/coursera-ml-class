function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% ignore the first element of theta. As said on page 4 it should
% not be used for regularization
theta_bis = [0 ; theta(2:size(theta), :)];

J =  1/(2*m) * sum(((X * theta) -y ) .^2);
J = J + (lambda * sum(theta_bis .^ 2)) / (2*m);

% reuse theta_bis - see gradient function, the first
% element of theta is not used in the second term.
grad = ( X' * ((X * theta)- y) + lambda*theta_bis ) / m;


% =========================================================================

grad = grad(:);

end
