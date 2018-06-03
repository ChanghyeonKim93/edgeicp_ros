#ifndef _EDGEICP_H_
#define _EDGEICP_H_

#include <ros/ros.h>

#include <iostream>
#include <sys/time.h>
#include <ctime>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#define PI 3.141592

typedef std::string TopicTime;

class Edgeicp{
public:
  struct Calibration {
    double fx;  //  focal length x
    double fy;  //  focal length y
    double cx;  //  principal point (u-coordinate)
    double cy;  //  principal point (v-coordinate)
    int width ;
    int height;

    Calibration () {
      fx     = 620.608832234754;
      fy     = 619.113993685335;
      cx     = 323.902900972212;
      cy     = 212.418428046497;
      width  = 752;
      height = 480;
    }
  };


  struct Hyperparameters {
    int nSample;
	  int maxIter;
	  int shiftIter;
    double treeDistThres;
	  double transThres;
	  double rotThres;

    Hyperparameters () {
      nSample       = 500;  // the number of sampling pixels from the current frame.
    	maxIter       = 30;   // Maximum iteration number for an image.
      shiftIter     = 7;    // After 7 iterations, shift to the 2d NN searching.heuristic.
    	treeDistThres = 15.0; // pixels, not use
    	transThres    = 0.05; // 2 cm
    	rotThres      = 3;    // 3 degree
    }
  };


  // debug parameters
  struct Debug {
    bool imgShowFlag;

    Debug () {
      imgShowFlag = false;
    }
  };


  struct Parameters {
    Edgeicp::Calibration     calib;
    Edgeicp::Debug           debug;
    Edgeicp::Hyperparameters hyper;
  };

public: // Methods
  Edgeicp(Parameters params_); // constructor
  ~Edgeicp();                  // desctructor.
  void run();                  // one cycle of the algorithm.
  void image_acquisition(const cv::Mat& img, const cv::Mat& depth, const TopicTime& curTime_);
  void getMotion(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw);

public: // Public variables
  bool completeFlag;

private: // Private variables
  Parameters params;   // For parameters which can be defined by the user.
  TopicTime  curTime;  // current time. (std::string)
  TopicTime  prevTime; // previuos time. (std::string)

  cv::Mat curImg,  curDepth; // current image data
  cv::Mat keyImg,  keyDepth; // keyframe image data

  bool isInit;           // boolean indicating whether it is the first iteration or not.

  Eigen::MatrixXd tmpXi; // The se(3) from the current keyframe to the current frame.
  Eigen::MatrixXd delXi; // infinitisimal motion update during the optimization.
};
#endif
