#include "random_function.h"

void rnd::randsample(const int& nPoints, const int& nSample, std::vector<int>& rndIdx_) { //sampling without replacement
  rndIdx_.reserve(0); // initialize
  if(nPoints <= nSample)  rndIdx_.resize(nPoints, 0);
  else rndIdx_.resize(nSample, 0);

  std::vector<int> fullIdx;
  fullIdx.reserve(0);
  fullIdx.resize(nPoints, 0);

  int i = 0;
  for(std::vector<int>::iterator it = fullIdx.begin(); it != fullIdx.end(); it++, i++) *it = i;

  std::random_shuffle(fullIdx.begin(), fullIdx.end());
  //sampling specific number from fullIdx
  if(nPoints <= nSample) {
    for(i = 0; i < nPoints; i++) rndIdx_[i] = fullIdx[i];
  }
  else{
    for(i = 0; i < nSample; i++) rndIdx_[i] = fullIdx[i];
  }
};

//TODO: randn function.
double rnd::randn(){
  return -1;
};
