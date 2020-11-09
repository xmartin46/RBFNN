// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

#include <math.h>

struct parDistMat : public Worker {
  
  // input matrix to read from
  const RMatrix<double> dataset;
  const RMatrix<double> centroids;
  const double std;
  const std::string dataset_type;
  
  // output matrix to write to
  RMatrix<double> rmat;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  parDistMat(const NumericMatrix& dataset, const NumericMatrix& centroids, const double std, const std::string dataset_type, NumericMatrix rmat)
    : dataset(dataset), centroids(centroids), std(std), dataset_type(dataset_type), rmat(rmat) {}
  
  // function call operator that work for the specified range (begin/end)
  void operator()(std::size_t begin, std::size_t end) {
    if (dataset.ncol() != centroids.ncol()) {
      throw std::runtime_error("Incompatible number of dimensions");
    }
    
    if (dataset_type != "original_dataset" and dataset_type != "distance_matrix" and dataset_type != "kernel_matrix") {
      throw std::runtime_error("Dataset type is not correct!");
    }
    
    for (std::size_t i = begin; i < end; i++) {
      
      RMatrix<double>::Row dataRow = dataset.row(i);
      
      for (std::size_t j = 0; j < centroids.nrow(); j++) {
        
        RMatrix<double>::Row centRow = centroids.row(j);
        
        double aux = 0;
        
        if (dataset_type != "kernel_matrix") {
          if (dataset_type == "original_dataset") {
            for (std::size_t k = 0; k < centroids.ncol(); k++) {
              aux += pow(dataRow[k] - centRow[k], 2.0);
            }
          } else {
            aux = centRow[i];
          }
          
          aux = exp(-aux/(2 * pow(std, 2)));
        } else {
          aux = centRow[i];
        }
        
        rmat(i, j) = aux;
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix rcpp_parallel_kernel_matrix(NumericMatrix& dataset, NumericMatrix& centroids, double std, std::string dataset_type) {
  
  // allocate the matrix we will return
  NumericMatrix rmat(dataset.nrow(), centroids.nrow());
  
  // create the worker
  parDistMat parDistMat(dataset, centroids, std, dataset_type, rmat);
  
  // call it with parallelFor
  parallelFor(0, dataset.nrow(), parDistMat);
  
  return rmat;
}