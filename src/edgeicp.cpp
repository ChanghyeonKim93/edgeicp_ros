#include "edgeicp.h"

#include <iostream>
#include <sys/time.h>


Edgeicp::Edgeicp(Parameters params_) : params(params_) {
  if(params.debug.imgShowFlag == true) cv::namedWindow("current image");
  this->completeFlag = false;
  this->isInit     = true;
}

Edgeicp::~Edgeicp() {
  ROS_INFO_STREAM("Edgeicp node is terminated.\n");
}

void Edgeicp::image_acquisition(const cv::Mat& img_, const cv::Mat& depth_, const TopicTime& curTime_){
  this->curImg    = img_.clone();
  this->curDepth  = depth_.clone();
  this->curTime   = curTime_;
  ROS_INFO_STREAM("In algorithm image is updated.");
}

void Edgeicp::run() {
  this->completeFlag = true; //

  if(this->isInit == true){ // keyframe initialization
    ROS_INFO_STREAM("FIRST ITERATION - keyframe initialize");
    this->tmpXi  = Eigen::MatrixXd::Zero(6,1);
    this->delXi  = Eigen::MatrixXd::Zero(6,1);

    this->keyImg   = this->curImg;
    this->keyDepth = this->curDepth;

    //
    this->isInit = false;
  }
  else{ // After initial images




    if(0){ // if the distance from the current keyframe to the current frame exceeds the threshold, renew the key frame
      this->keyImg   = this->curImg;
      this->keyDepth = this->curDepth;


    }
  }

  if(this->params.debug.imgShowFlag == true){
    cv::imshow("current image", this->curImg);
    cv::waitKey(5);
  }
}

void Edgeicp::getMotion(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw){

}
