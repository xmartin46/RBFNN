###############################################################################  
##                                                                           ##  
## RBFNN - Radial Basis Function Neural Networks                             ##  
##                                                                           ##  
## Author: Xavier Martín     xmartin46@gmail.com                             ##
##                                                                           ##   
###############################################################################  

hiddenLayer <- function(X, centroids, stds, inputType="original_dataset") {
  ## Compute the interpolation matrix given the dataset, the centroids in the hidden layer and the standard deviations
  #### Depending on the type of the input dataset (original, distances or kernel), the Rcpp function will
  #### do less or more computations.
  interpolateMatrix <- rcpp_parallel_kernel_matrix(as.matrix(X), as.matrix(centroids), stds[1], inputType)
  
  interpolateMatrix <- as.data.frame(interpolateMatrix)
  interpolateMatrix
}