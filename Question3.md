## Question 3

In logistic regression, the computational cost for each gradient descent iteration is primarily determined by the cost of calculating the gradient of the cost function with respect to the model parameters. This involves matrix operations, specifically the multiplication of the feature matrix with the vector of parameter estimates.

Given:

m - training examples

n -  features

Each training example has an average of k non-zero entries, and k<<n (indicating sparsity)

The computational complexity can be approximated as follows:

1. Gradient Calculation: The gradient of the cost function in logistic regression is computed as the product of the transpose of the feature matrix (size: m xn) and the error vector (size: m x 1). However, due to the sparsity of the matrix, each row has only about k non-zero entries on average.

2. Sparse Matrix Operations: Modern well-written packages for logistic regression utilize sparse matrix representations and operations. This means that only non-zero elements are stored and involved in computations.

3. Approximate Cost: For each example, we are effectively performing k multiplications(since only k out of n features contribute). This needs to be done for all examples.Therefore, the cost per gradient descent iteration is approximately O(m x k ).

4. Ignoring Lower Order Terms: Since k << n the cost of operations that scale with n (like adding the regularization term, if present) can often be ignored in the approximation.

5. Constant Factors: The actual constant factors depend on the specific implementationdetails of the logistic regression algorithm in the package, like how it handles sparsedata structures, any optimizations used, etc.

Thus, in a modern, well-written package that efficiently handles sparse matrices, theapproximate computational cost of each gradient descent iteration in logistic regression is O(m x k) This is significantly lower than the 
O(m x n) cost we would expect with dense matrices, especially in cases where n >> k.
