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

/*
void Edgeicp::find_valid_mask(const cv::Mat& edgeMask, const cv::Mat& gradXMap, const cv::Mat& gradYMap, const cv::Mat& validMask, const int& numValidPx){
  cv::Mat tmpMat;
  numValidPx = 0; // initialize
  tmpMat.create(edgeMask.size(),CV_8U);

  int u=0, v=0;
  for(v = 0; v < edgeMask.rows; v++){
    uchar  *edgePtr  = edgeMask.ptr<uchar>(v);
    double *depthPtr = depth_map.ptr<double>(v);
    uchar  *tmpPtr   = tmpMat.ptr<uchar>(v);

    for(u = 0; u < edgeMask.cols; u++){
      if(*(edgePtr++) > 0 & *(depthPtr++) > 0){
        *(tmpPtr++) = 255;
        numValidPx++;
      }
      else{
        *(tmpPtr++) = 0;
      }
    }
  }
  tmpMat.copyTo(img_o); // output : valid pixel mask cv::Mat image
}
*/


void Edgeicp::run() {
  this->completeFlag = true; //

  if(this->isInit == true){ // keyframe initialization
    ROS_INFO_STREAM("FIRST ITERATION - keyframe initialize");
    this->tmpXi  = Eigen::MatrixXd::Zero(6,1);
    this->delXi  = Eigen::MatrixXd::Zero(6,1);

    // Initialize the keyImg and keyDepth.
    this->keyImg   = this->curImg;
    this->keyDepth = this->curDepth;

    // Calculate the gradient values along each direction.
    //cv::Sobel(this->keyImg, dx_short, CV_16S, 1,0,3,1,0, cv::BORDER_DEFAULT); // CV_16S : short -32768~32768, CV_64F : double
    //cv::Sobel(this->keyImg, dy_short, CV_16S, 0,1,3,1,0, cv::BORDER_DEFAULT);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->keyImg, this->keyEdgeMap, 170,220);
    this->curEdgeMap = this->keyEdgeMap;
    // TODO : At this location, kd tree construction.


    this->isInit = false;
  }
  else{ // After initial images, successively run the algorithm for the current image.

    // Find edge region
    //cv::Sobel(this->curImg, dx_short, CV_16S, 1,0,3,1,0,cv::BORDER_DEFAULT); // CV_16S : short -32768~32768, CV_64F : double
    //cv::Sobel(this->curImg, dy_short, CV_16S, 0,1,3,1,0,cv::BORDER_DEFAULT);
    //cv::GaussianBlur(dx_short,dx_short,cv::Size(3,3),0,0);
    //cv::GaussianBlur(dy_short,dy_short,cv::Size(3,3),0,0);
    //RGBDIMAGE::calcDerivNorm(dx_short, dy_short, d_norm, dx, dy);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->curImg, this->curEdgeMap, 170,220);

    if(0){ // if the distance from the current keyframe to the current frame exceeds the threshold, renew the key frame
      this->keyImg   = this->curImg;
      this->keyDepth = this->curDepth;


    }
  }

  if(this->params.debug.imgShowFlag == true){
    cv::imshow("current image", this->curEdgeMap);
    cv::waitKey(5);
  }
}

void Edgeicp::getMotion(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw){

}
