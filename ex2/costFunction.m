function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values

J = 0;
m = length(y); % number of training examples
m_theta = length(theta); 
grad = zeros(size(theta));

tmp = 1;
total = 0;
while tmp <= m,
    current = 0;
    s = sigmoid ((theta') * X(tmp));
    current = -1 * y(tmp) * log(s) - ( 1 - y(tmp)) * log (1 -  s );

    total = total + current;
    tmp = tmp + 1;
end

J = total / m;

s = sigmoid(X*theta);
J = (1/m) * ((-y' * log(s)) - (( 1 - y)' * log(1-s)));

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
s = sigmoid(X*theta);
grad = (1 / m) * ((s - y)' * X);



% =============================================================

end
