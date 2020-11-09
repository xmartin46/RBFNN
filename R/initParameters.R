###############################################################################  
##                                                                           ##  
## RBFNN - Radial Basis Function Neural Networks                             ##  
##                                                                           ##  
## Author: Xavier Martín     xmartin46@gmail.com                             ##
##                                                                           ##   
###############################################################################  

#' @importFrom parallelDist parDist
initParameters <- function(X, neurons, it, inputType="original_dataset") {
  dataset <- as.data.frame(X)
  
  ## Take #instances = neurons in a random manner.
  seed <- it * ncol(dataset)
  set.seed(seed)
  centroid_indexes <- sample(nrow(dataset), neurons)
  
  if (inputType != "original_dataset" && inputType != "distance_matrix" && inputType != "kernel_matrix") {
    warning('Input type is not in any of the correct formats!')
  }
  
  ## Compute the standard deviations of the clusters
  if (inputType != "kernel_matrix") {
    if (inputType == "original_dataset") {
      centroids <- as.matrix(dataset[centroid_indexes, ])
      centroid_distances <- as.matrix(parDist(centroids))
      maxim <- max(centroid_distances)
    } else if (inputType == "distance_matrix") {
      centroid_distances <- sqrt(dataset[centroid_indexes, centroid_indexes])
      maxim <- max(centroid_distances)
    }
    
    # stds <- data.frame()
    # p <- 2
    # for (i in 1:length(centroid_indexes)) {
    #   aux <- sort(centroid_distances[i, ] ^ 2)
    #   aux2 <- sqrt(sum(aux[1:(p+1)]))/p
    #   stds <- rbind(stds, aux2)
    # }
    # stds <- as.matrix(stds)
    
    stds <- rep(maxim/(sqrt(2 * neurons)), each=neurons)
    # print(stds)
    # soptif
  } else {
    stds <- Inf
  }
  
  ## Return the parameters
  newList <- list("centroids" = dataset[centroid_indexes, ], "centroid_indexes" = centroid_indexes, "stds" = stds)
  newList
}