# Radial Basis Function Neural Networks

Radial Basis Function Neural Network (RBFNN) is a particular
type of Artificial Neural Network (ANN). The difference with other types
of ANNs is that they use radial basis functions as the activation
function\[1\]. It has been used in a wide variety of fields including
classification, interpolation, time-series analysis and image
processing.

# Network architecture

Theoretically, RBFNNs can be employed in any model and network
(single-layer or multi-layer). However, since , the traditional
architecture (Figure [1](#fig:RBFN)) consists of three layers: one input
layer, one hidden layer and one output layer.

![Radial Basis Function Neural Network
architecture.](./images/RBFN2.png)

The input vector \(\bold{x}\) is a \(n\)-dimensional array that is
forwarded to each neuron in the hidden layer. Each neuron \(i\) in the
hidden layer has an instance of the training set (usually called
centroid or *prototype*), and computes an RBF as its nonlinear
activation function, typically the Gaussian\[2\]:

\[\label{eq:rbf}
    h_i(\bold{x}) = \exp\left[- \dfrac{||\bold{x} - \bold{c}_i||^2}{2\sigma_i^2}\right]\]

where \(c_i\) is the centroid of the neuron \(i\) and \(h_i\) its
output. Usually, the same RBF is applied to all neurons. The outputs of
the neurons are linearly combined with weights
\(\bold{w} = \{w_i\}^k_{i = 1}\) to produce the network output
\(f_{t}(\bold{x})\):

\[f_{t}(\bold{x}) = \sum_{i = 1}^k w_i h_i(\bold{x})
    \label{eq:RBFN_output}\]

It can be included an additional neuron in the hidden layer to model the
biases of the output layer. This neuron does not have any centroid but a
constant activation function \(h_{0}(\bold{x}) = 1\).

Depending on the type of problem we are addressing, the output layer
will vary its number of neurons. If it is a regression task, there is
needed only 1 neuron. For binary classification, it is also needed only
1 neuron (applying logistic regression). For classification involving
more than 2 classes (multinomial logistic regression), we need to use
more than 1 neuron (specifically, one for each class) and apply the
softmax function to normalize the output values for later assigning the
input vector\(\bold{x}\) the class of the output with maximum value.

\[Class(\bold{x}) = \argmax_{t \in \{1, \dots, C\}}f_{t}(\bold{x})\]

# Learning phase

As explained earlier, the parameters of this neural network are the
centroids \(c_{i}\) and the width \(\sigma_{i}\) of each RBF, and the
weights from the hidden layer to the output layer. There are multiple
ways of training each parameter. Some produce better results than others
but are more costly. It is up to the user to decide where to find the
balance between time efficiency and accuracy.

The learning is usually performed in a two-phase strategy. The first one
focuses on the hidden layer, selecting suitable centers and their
respective width, whereas the second phase focuses on the output layer
adjusting the network weights.

## Selecting RBF centroids

In order to select the centroids for the RBFNN, typically unsupervised
training methods from clustering are used, such as the \(k\)-means
clustering algorithm.

The \(k\)-means algorithm was selected as one of the top-10 data mining
algorithms identified by the International Conference on Data Mining
(ICDM) in December 2006 . It is a simple iterative method to partition a
given dataset into a specified number of clusters, \(k\). We must remark
that the value of \(k\) is crucial, as with too few centers the network
will not make a good generalization (underfitting) and with too many
centers the network will learn useless information from noisy data
(overfitting). This algorithm iterates between two steps until
convergence. First, it partitions the data, assigning each data point to
its nearest centroid. Second, each cluster representative (new centroid)
is relocated to the mean of all data points assigned to it. This process
is repeated until the assignment in step 1 does not vary. The metric
used is the Euclidean distance, and the objective function we are trying
to minimize is

\[J(\bold{x}) = \sum_{j = 1}^{k}\sum_{i = 1}^{n} ||\bm{x}_{i} - \bm{c}_{j}||^{2}\]

Convergence is guaranteed in a finite number of iterations. The
selection of the initial centroids is very important because the final
result depends on it. Although random initialization usually leads to
good results, there exists other initialization (e.g. \(k\)-means++)
which perform better.

Another way to select the centroids is by selecting randomly a subset of
the training dataset. It has one convenient, the training set must be
representative of the whole dataset. On the contrary case, learning
based on the random selection of the RBF centers may lead to poor
performances.

## Selecting RBF widths

The setting of the RBF widths is also crucial for the network. When they
are too large, the estimated probability density is over-smoothed and
the estimated true probability density may be lost. On the other hand,
when it is too small, there may be overfitting to the particular
training set.

There is a wide variety of methods for selecting the widths based on
heuristics. Following are explained the most commonly used. If we have
the clusters pre-computed, we can compute the width of each RBF by just
computing the standard deviation of each cluster
\(\sqrt{\sum_{x \in cluster \hspace{0.05cm} i} ||\bm{x} - \bm{c}_{i}||^{2}/n_{i}}\).
Alternatively, the width \(\sigma_{i}\) can be the average of the
distances between the \(i\)-th RBF centroid and its \(L\) nearest
centroids or the distance between its closest centroid multiplied by a
factor \(a\) ranging from 1.0 to 1.5
(\(\sigma_{i} = a||\bm{c}_{i} - \bm{c}_{closest \hspace{0.05cm} to \hspace{0.05cm} \bm{c}_{i}}||\)).

Moreover, using the same width for all RBF is proven to have universal
approximation capability. It can be selected as the
\(d_{max}/\sqrt{2k}\), where \(d_{max}\) is the maximum distance between
the selected centroids. This choice makes the RBF neither too steep nor
too flat.

## Training the output weights

For regression problems, the output layer is just a linear combination
of the outputs from the RBFs (Eq.
[\[eq:RBFN\_output\]](#eq:RBFN_output)). Hence, the weights that
minimize the error at the output can be computed with a linear
pseudo-inverse solution:

\[\label{eq:LSLR}
    \bold{W} = \bm{X}^{+}y = (\bm{X}^{T}X)^{-1}\bm{X}^{T}y\]

where \(X\) is the matrix of the values of the RBFs and \(y\) the labels
of the dataset. Using the Least Squares Linear Regression equation we
ensure a global minimum and the cost function is relatively fast.

For classification problems, though, the use of linear outputs is less
appropriate because the outputs would not represent the class
probabilities. For binary or multi-class classification problems the
outputs can be trained efficiently using algorithms derived from
Generalised Linear Models (GLM). The obvious starting point for training
these models is to use maximum likelihood. For linear regression, this
is the same as using the Least Squares Linear Regression (Eq.
[\[eq:LSLR\]](#eq:LSLR)). We can see, then, that this equation is
derived from GLM. However, we are working with binary and multi-class
classification problems, and the maximum likelihood does not lead to a
quadratic form. Hence, we need an iterative method that makes use of
GLM. The algorithm used is called Iteratively Reweighted Least Squares
(IRLS). It makes use of the Hessian matrix. As it is positive
semi-definite, it is held that there is a unique maximum. Consequently,
it is included in the convex programming field, therefore it can be used
with the Newton-Raphson algorithm. See  for more information about GLM
and IRLS.

Another possible training algorithm is gradient descent, a supervised
learning algorithm. In gradient descent training, the weights are
adjusted at each time step by moving them in a direction opposite to the
gradient of the objective function (thus allowing the minimum of the
objective function to be found). This method, however, is
computationally very expensive and usually takes time to converge. In
principle, it is better to take advantage of the special *near-linear*
form of the RBFNN model and use the GLM method instead of gradient
descent.

After this final step of calculating the output layer weights, all
parameters of the RBF network have been determined.

# RBFNNs versus MLPs

RBFNNs differ from Multilayer Perceptron (MLP) architecture in some
aspects. A comparison between MLP and RBFNN is described below.

The MLP has a very complex error surface, which hinders the problem of
local minima or nearly flat regions for the gradient descent algorithm.
Else way, the RBFNN has a simpler architecture with a linear
combination, which makes it easy to find the unique solution to the
weights by the Least Squares Linear Regression equation, equivalent to a
gradient descent of a quadratic surface (convex). Because of the flat
regions talked about earlier, the convergence in the training process
for MLP is slow. The RBFNN requires less training time than MLP trained
with the backpropagation rule.

Another difference is that MLP generalizes more for each training
sample. In the RBFNN, each RBF focuses on its neighborhood. As a result,
the RBFNN suffers from the curse of dimensionality, which translates to
needing much more hidden neurons or data than the MLP to achieve the
same results. According to , in order to approximate a wide class of
smooth functions, the number of hidden units required for a three-layer
MLP is polynomial w.r.t. the input dimensions, whereas in the RBF it is
exponential. Due to needing more hidden neurons, the MLP is much faster
in the feed-forward phase, when predicting.

Finally, the interpretation of each node in the hidden layer is easier
in RBFNNs.

1.  A radial basis function (RBF) is a real-valued function in which
    their response decreases (or increases) monotonically with distance
    from a central point.

2.  Other types of RBF are multiquadric, inverse multiquadric and Cauchy
    functions.
