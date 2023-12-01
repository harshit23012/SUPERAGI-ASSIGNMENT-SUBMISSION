### Question 1

In logistic regression, when we train a model with a set of features and learn the weights $w_{0}, w_{1},......, w_{n}$, each weight corresponds to the importance of its respective feature in predicting the target variable. If feature $n$ gets weight $w_{n}$, at the end of training, this weight represents how much feature a contributes to the outcome.

If we then create a new dataset by duplicating feature $n$ into feature $n + 1$ and retrain the model, we now have two identical features, each with their own weight in the new model, $w_{new_{n}}$, and  $w_{new_{n+1}}$.


The likely relationship between the new weights  $w_{new_{n}}$ and  $w_{new_{n+1}}$ will depend on the regularization used during training and the optimization algorithm's behavior.
However, in many cases, without regularization, the model may assign half the original weight wn to each of the duplicated features because the overall contribution to the prediction should stay roughly the same. This means:

 $w_{new_{n}} +  w_{new_{n+1}}  \approx    w_{n}$

However, regularization changes this dynamic:

L1 Regularization (Lasso): This type of regularization adds a penalty equivalent to the absolute value of the magnitude of coefficients. This can lead to some feature weights being reduced to zero, effectively removing them from the model. Because it tends to produce sparse solutions, with duplicated features, L1 regularization might zero out one of the duplicated feature weights while giving the entire weight to the other, or distribute it unevenly between them.

L2 Regularization (Ridge): L2 regularization adds a penalty equivalent to the square of the magnitude of coefficients. This discourages large weights in general but does not tend to zero out weights completely. With duplicated features, L2 regularization would likely result in both features receiving some weight, with the sum of their weights being close to the original weight wn. The distribution might not be exactly half, but the total influence of the duplicated features (the sum of their weights) would typically not be more than the influence of the original single feature..

Elastic Net Regularization: This is a combination of L1 and L2 regularization. It might distribute weights between the two duplicated features in a manner influenced by both L1 and L2 tendencies.

In real-world scenarios, other factors might influence the final weights, such as:

The optimization algorithm used (like gradient descent, stochastic gradient descent, etc.).
The convergence criteria of the algorithm.
The scale of the regularization penalty.
The presence of other features and their correlation with the duplicated feature.

In summary, the relationship between the new weights will depend on the regularization technique and other training parameters. The underlying concept is that the model will distribute weight between the duplicated features in a way that reflects their overall importance to the prediction task, which would have been represented by a single weight in the original model.
