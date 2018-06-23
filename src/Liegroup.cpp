#include "Liegroup.h"

void lie::se3_exp(const Eigen::MatrixXd& xi, Eigen::MatrixXd& g){
  // initialize variables
  Eigen::Vector3d v, w;
  double length_w = 0.0;
  Eigen::Matrix3d Wx, R, V;
  Eigen::Vector3d t;

  v(0) = xi(0);
  v(1) = xi(1);
  v(2) = xi(2);
  w(0) = xi(3);
  w(1) = xi(4);
  w(2) = xi(5);

  length_w = std::sqrt(w.transpose() * w);
  hat_operator(w, Wx);
  
  if (length_w < 1e-7) {
      R = Eigen::Matrix3d::Identity(3,3) + Wx + 0.5 * Wx * Wx;
      V = Eigen::Matrix3d::Identity(3,3) + 0.5 * Wx + Wx * Wx / 3.0;
  }
  else {
      R = Eigen::Matrix3d::Identity(3,3) + (sin(length_w)/length_w) * Wx + ((1-cos(length_w))/(length_w*length_w)) * (Wx*Wx);
      V = Eigen::Matrix3d::Identity(3,3) + ((1-cos(length_w))/(length_w*length_w)) * Wx + ((length_w-sin(length_w))/(length_w*length_w*length_w)) * (Wx*Wx);
  }
  t = V * v;

  // assign rigid body transformation matrix (in SE(3))
  g = Eigen::MatrixXd::Identity(4,4);
  g(0,0) = R(0,0);
  g(0,1) = R(0,1);
  g(0,2) = R(0,2);

  g(1,0) = R(1,0);
  g(1,1) = R(1,1);
  g(1,2) = R(1,2);

  g(2,0) = R(2,0);
  g(2,1) = R(2,1);
  g(2,2) = R(2,2);

  g(0,3) = t(0);
  g(1,3) = t(1);
  g(2,3) = t(2);

      // for debug
      // std::cout << R << std::endl;
      // std::cout << t << std::endl;
      //usleep(10000000);
}

void lie::hat_operator(const Eigen::Vector3d& colVec, Eigen::Matrix3d& skewMat){
   skewMat(0,0) = 0;
   skewMat(0,1) = -colVec(2);
   skewMat(0,2) = colVec(1);

   skewMat(1,0) = colVec(2);
   skewMat(1,1) = 0;
   skewMat(1,2) = -colVec(0);

   skewMat(2,0) = -colVec(1);
   skewMat(2,1) = colVec(0);
   skewMat(2,2) = 0;
}


void lie::a2r(const double& r, const double& p, const double& y, Eigen::Matrix3d& R){// r,p,y are defined on the radian domain.
  Eigen::Matrix3d Rx, Ry, Rz;

  Rx<<1,0,0,0,cos(r),sin(r),0,-sin(r),cos(r);
  Ry<<cos(p),0,-sin(p),0,1,0,sin(p),0,cos(p);
  Rz<<cos(y),sin(y),0,-sin(y),cos(y),0,0,0,1;
  R = Rz*Ry*Rx;
}
