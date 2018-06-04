#include "random_function.h"

void rnd::randsample(const int& nPoints, const int& nSample, std::vector<int>& refIdx) { //sampling without replacement
  refIdx.reserve(0); // initialize
  if(nPoints <= nSample)  refIdx.resize(nPoints, 0);
  else refIdx.resize(nSample, 0);

  std::vector<int> fullIdx;
  fullIdx.reserve(0);
  fullIdx.resize(nPoints, 0);

  int i = 0;
  for(std::vector<int>::iterator it = fullIdx.begin(); it != fullIdx.end(); it++, i++) *it = i;

  std::random_shuffle(fullIdx.begin(), fullIdx.end());
  //sampling specific number from fullIdx
  if(nPoints <= nSample) {
    for(i = 0; i < nPoints; i++) refIdx[i] = fullIdx[i];
  }
  else{
    for(i = 0; i < nSample; i++) refIdx[i] = fullIdx[i];
  }
};
