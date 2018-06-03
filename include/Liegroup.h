#ifndef _LIEGROUP_H_
#define _LIEGROUP_H_
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

namespace lie{
  void se3_exp(const Eigen::MatrixXd& xi, Eigen::Matrix4d& g);
  void hat_operator(const Eigen::Vector3d& colVec, Eigen::Matrix3d& skewMat);
};
#endif
