#define _ROBUST_WEIGHTING_H_
#ifndef _ROBUST_WEIGHTING_H_

#include <iostream>
#include <cmath>

namespace rweight{
  void update_t_distribution(const std::vector<double>& residual, double& sigma);
};

#endif
