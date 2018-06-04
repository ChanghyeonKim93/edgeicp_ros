#include "robust_weighting.h"

void rweight::update_t_distribution(const std::vector<double>& residual, double& sigma){
  double nu = 2.0;
  int N = residual.size();
  double lambda_prev = 1.0/(sigma*sigma);
  double temp=0.0, lambda_curr=0.0, sum=0.0;
  double eps = 0.0000001;
  while(1){
    for(int i=0;i<N;i++) sum+= residual[i]*residual[i] / ( nu + lambda_prev*residual[i]*residual[i]);
    temp = ( (nu+1.0)/(double)N )*sum;

    if(fabs(lambda_curr-lambda_prev)<=eps) break;
    lambda_prev = lambda_curr;
  }
  sigma = sqrt(1.0/lambda_prev);
}
