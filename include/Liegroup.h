#ifndef _LIEGROUP_H_
#define _LIEGROUP_H_
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

namespace lie{
  void se3_exp(const Eigen::MatrixXd& xi, Eigen::MatrixXd& g);
  void se3_log(const Eigen::Matrix4d& g, Eigen::MatrixXd& xi);

  void a2r(const double& r, const double& p, const double& y, Eigen::Matrix3d& R);


  void hat_operator(const Eigen::Vector3d& colVec, Eigen::Matrix3d& skewMat);
};
#endif
