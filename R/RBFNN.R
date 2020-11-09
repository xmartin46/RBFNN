###############################################################################  
##                                                                           ##  
## RBFNN - Radial Basis Function Neural Networks                             ##  
##                                                                           ##  
## Author: Xavier Martín     xmartin46@gmail.com                             ##
##                                                                           ##   
###############################################################################  

#' @title Train a Radial Basis Function Neural Network.
#' @name RBFNN_train
#' 
#' @param X original dataset, distance matrix or gaussian kernel matrix.
#' @param Y the labels of \code{X}. Each observation has one label.
#' @param neurons number of neurons in the hidden layer.
#' @param it value used in the seed.
#' @param inputType type of \code{X}. Available types: original_dataset (original dataset), distance_matrix (Euclidean distance matrix ) and kernel_matrix (Gaussian kernel matrix).
#' 
#' @return The trained model, the centroids of the RBFNN, the standard deviations of each RBF neuron and indexes of the observations that are the centroids.
#' @export
#' 
#' @importFrom nnet multinom
#' 
#' @examples
#' data(iris)
#' dataset <- iris[, 1:4]
#' labels <- iris[, 5]
#' 
#' X_train <- dataset[1:100, ]
#' y_train <- labels[1:100]
#' X_test <- dataset[101:150, ]
#' y_test <- labels[101:150]
#' 
#' neurons <- 10
#' seed_val <- 1
#' RBFNN <- RBFNN_train(as.matrix(X_train), as.matrix(y_train), neurons, seed_val)
RBFNN_train <- function(X, Y, neurons, it, inputType="original_dataset") {
  ## Compute initial parameters: centroid indexes and standard deviations of clusters
  params <- initParameters(X, neurons, it, inputType = inputType)
  centroid_indexes <- params$centroid_indexes
  centroids <- params$centroids
  stds <- params$stds
  
  ## Compute the interpolate matrix using the input dataset, the centroids in the hidden layer and the standard deviations
  interpolateMatrix <- hiddenLayer(X, centroids, stds, inputType = inputType)
  
  ## Group input and labels
  interpolateMatrix <- cbind(interpolateMatrix, Y)
  interpolateMatrix <- as.data.frame(interpolateMatrix)
  
  ## Rename the interpolate matrix (for multinom predict function needs)
  colnames(interpolateMatrix) <- paste("col", 1:neurons, sep="")
  names(interpolateMatrix)[neurons + 1] <- paste("label")
  
  ## Apply multinomial logistic regression
  model <- multinom(label ~ ., data = interpolateMatrix, maxit=100, MaxNWts=5000, trace=FALSE)
  
  ## Return the model and the parameters
  if (inputType == "distance_matrix" || inputType == "kernel_matrix") {
    centroid_indexes_in_complete_distance_matrix <- as.numeric(row.names(centroids))
  } else {
    centroid_indexes_in_complete_distance_matrix <- NULL
  }
  
  newList <- list("model" = model, "centroids" = centroids, "centroid_indexes_in_complete_distance_matrix" = centroid_indexes_in_complete_distance_matrix, "stds" = stds)
  newList
}


#' @title Predict using a Radial Basis Function Neural Network.
#' @name RBFNN_predict
#' 
#' @param model The trained RBFNN.
#' @param X original dataset, distance matrix or gaussian kernel matrix.
#' @param centroids centroids of the model.
#' @param stds standard deviations of each RBF.
#' @param inputType type of \code{X}. Available types: original_dataset (original dataset), distance_matrix (Euclidean distance matrix ) and kernel_matrix (Gaussian kernel matrix).
#' 
#' @return The predicted labels (in classification) or the value (in regression).
#' @export
#' 
#' @importFrom stats predict
#' 
#' @examples
#' data(iris)
#' iris <- iris
#' ## Random permutation because the labels are ordered and therefore the accuracy would not
#' ## be correct (only for this dataset).
#' iris <- iris[sample(nrow(iris)), ]
#' 
#' dataset <- iris[, 1:4]
#' labels <- iris[, 5]
#' 
#' X_train <- dataset[1:100, ]
#' y_train <- labels[1:100]
#' X_test <- dataset[101:150, ]
#' y_test <- labels[101:150]
#' 
#' neurons <- 10
#' seed_val <- 1
#' RBFNN <- RBFNN_train(as.matrix(X_train), as.matrix(y_train), neurons, seed_val)
#' 
#' model <- RBFNN$model
#' centroids <- RBFNN$centroids
#' stds <- RBFNN$stds
#' 
#' predictions <- RBFNN_predict(model, X_test, centroids, stds)
#' table(predictions, y_test)
#' accuracy <- sum(diag(table(predictions, y_test)))/nrow(X_test)
RBFNN_predict <- function(model, X, centroids, stds, inputType = "original_dataset") {
  ## Compute the interpolate matrix using the test dataset, the centroids of the RBFNN in the hidden layer and the standard deviations
  interpolateMatrix <- hiddenLayer(X, centroids, stds, inputType = inputType)
  interpolateMatrix <- as.data.frame(interpolateMatrix)
  
  ## Rename the interpolate matrix (for multinom predict function needs)
  centroids <- as.data.frame(centroids)
  colnames(interpolateMatrix) <- paste("col", 1:nrow(centroids), sep="")
  
  ## Return predictions
  predict(model, interpolateMatrix)
}