#define _RANDOM_FUNCTION_H_
#ifdef _RANDOM_FUNCTION_H_

#include <iostream>
#include <algorithm> // shuffle function
//#include <random>
#include <vector>
//#include <cmath>

namespace rnd{
  void randsample(const int& nPoints, const int& nSamples, std::vector<int>& rndIdx_);
  double randn(); // not yet done.
};

#endif
