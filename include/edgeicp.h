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

#include "KDTree.h"
#include "random_function.h"
#include "Liegroup.h"
#include "random_function.h"

#define PI 3.141592

typedef std::string TopicTime;

class Edgeicp{

public:
  struct Calibration {
    double fx;    //  focal length x
    double fy;    //  focal length y
    double invFx; // inverse value of focal length x
    double invFy; // inverse value of focal length y
    double cx;    //  principal point (u-coordinate)
    double cy;    //  principal point (v-coordinate)
    double depthScale;
    double width ;
    double height;

    Calibration () {
      fx          = 535.4/2.0;
      fy          = 539.2/2.0;
      invFx       = 1.0 / fx;
      invFy       = 1.0 / fy;
      cx          = (320.1+0.5)/2.0 - 0.5;
      cy          = (247.6+0.5)/2.0 - 0.5;
      depthScale  = 1.0;
      width       = 640.0/2.0;
      height      = 480.0/2.0;
    }
  };


  struct Hyperparameters {
    double treeDistThres;
	  double transThres;
	  double rotThres;
    double tDistNu;
    int nSample;
	  int maxIter;
	  int shiftIter;

    Hyperparameters () {
      nSample       = 500;  // the number of sampling pixels from the current frame.
    	maxIter       = 10;   // Maximum iteration number for an image.
      shiftIter     = 7;    // After 7 iterations, shift to the 2d NN searching.heuristic.
    	treeDistThres = 15.0; // pixels, not use
    	transThres    = 0.05; // 2 cm
    	rotThres      = 3.0;  // 3 degree
      tDistNu       = 2.0;  // t-distribution DOF = 2
    }
  };

  struct Cannyparameters {
    int lowThres;
    int highThres;

    Cannyparameters () {
      lowThres      = 60;
      highThres     = 150;
    }
  };


  // debug parameters
  struct Debug {
    bool imgShowFlag;
    bool textShowFlag;

    Debug () {
      imgShowFlag  = false; // default setting : do not show the debug images.
      textShowFlag = false;
    }
  };


  struct Parameters {
    Edgeicp::Calibration     calib;
    Edgeicp::Debug           debug;
    Edgeicp::Hyperparameters hyper;
    Edgeicp::Cannyparameters canny;
  };


private: // related to PixelData
  typedef struct PixelData_ { // This structure contains pixel data (u,v,depth,gy,gx).
    double u;
    double v;
    double d;
    double X;
    double Y;
    double Z;
    double gx;
    double gy;

    PixelData_(double u_, double v_, double d_, double X_, double Y_, double Z_, double gx_, double gy_) {
      u  = u_;
      v  = v_;
      d  = d_;
      X  = X_;
      Y  = Y_;
      Z  = Z_;
      gx = gx_;
      gy = gy_;
    }
  } PixelData;
  void delete_pixeldata(std::vector<PixelData*>& pixelDataVec);
  void initialize_pixeldata(const std::vector<PixelData*>& inputPixelDataVec_, std::vector<PixelData*>& outputPixelDataVec_);



public: // Methods used in main_script.cpp
  Edgeicp(Parameters params_); // constructor
  ~Edgeicp();                  // desctructor.
  void run();                  // one cycle of the algorithm.
  void set_images(const cv::Mat& img, const cv::Mat& depth, const TopicTime& curTime_);
  void get_motion(double& x, double& y, double& z, double& roll, double& pitch, double& yaw);



private: // Methods used in the algorithm privately.
  void downsample_image(const cv::Mat& imgInput, cv::Mat& imgOutput);
  void downsample_depth(const cv::Mat& imgInput, cv::Mat& imgOutput);
  void validify_depth(cv::Mat& imgInput);
  void calc_gradient(const cv::Mat& imgInput, cv::Mat& imgGradx, cv::Mat& imgGrady, cv::Mat& imgGrad, const bool& doGaussian);
  void find_valid_mask(const cv::Mat& imgInputEdge, const cv::Mat& imgDepth, const cv::Mat& imgGrad, cv::Mat& imgOutputEdge);
  void set_edge_pixels(const cv::Mat& imgInputEdge, const cv::Mat& imgDepth, const cv::Mat& imgGradx, const cv::Mat& imgGrady, const cv::Mat& imgGrad, std::vector<Edgeicp::PixelData*>& pixelDataVec);
  void calc_icp_residual_div(const std::vector<PixelData*>& curPixelDataVec_, const std::vector<PixelData*>& keyPixelDataVec_, const std::vector<int>& rndIdx_, const std::vector<int>& refIdx_, Eigen::MatrixXd& residual_);
  void warp_pixel_points(const std::vector<PixelData*> inputPixelDataVec_, const Eigen::MatrixXd& tmpXi_, std::vector<PixelData*>& warpedPixelDataVec_);
  void convert_pixeldatavec_to_vecvec2d(const std::vector<PixelData*>& pixelDataVec_, const std::vector<int>& indVec_, std::vector<std::vector<double>>& tmpPixel2Vec_);
  void convert_pixeldatavec_to_vecvec4d(const std::vector<PixelData*>& pixelDataVec_, const std::vector<int>& indVec_, std::vector<std::vector<double>>& tmpPixel4Vec_);
  void calc_icp_Jacobian_div(const std::vector<PixelData*>& warpedCurPixelDataVec_, const std::vector<PixelData*>& keyPixelDataVec_, const std::vector<int>& rndIdx_, const std::vector<int>& refIdx_, Eigen::MatrixXd& J_);
  void update_weight_matrix(const Eigen::MatrixXd& residual_, const double& sigma_, const double& nu_, Eigen::MatrixXd& W_);
  void multiply_weight_matrix(const Eigen::MatrixXd& J_, const Eigen::MatrixXd& W_, Eigen::MatrixXd& JW_);
  double mean_residual(const Eigen::MatrixXd& residual_);
  double update_t_distribution(const Eigen::MatrixXd& residual_, const double& sigma_, const double& nu_);



private: // Scripts



public: // Public variables
  bool completeFlag;


private: // Private variables
  Parameters params;   // For parameters which can be defined by the user.

  TopicTime  curTime;  // current time. (std::string)
  TopicTime  prevTime; // previuos time. (std::string)

  int numOfImg;

  // Images
  cv::Mat curImg,      curDepth; // current image data , Img : CV_8UC1 (datatype 1, uchar), depth : CV_16UC1 (datatype 2, ushort)
  cv::Mat keyImg,      keyDepth; // keyframe image data

  cv::Mat curImgLow,   curDepthLow;
  cv::Mat keyImgLow,   keyDepthLow;

  cv::Mat curEdgeMap,  curEdgeMapValid;
  cv::Mat keyEdgeMap,  keyEdgeMapValid;

  cv::Mat curImgGradx, curImgGrady, curImgGrad;
  cv::Mat keyImgGradx, keyImgGrady, keyImgGrad;

  cv::Mat debugImg, debugEdgeImg;


  // Pixel information containers.
  std::vector<PixelData*> curPixelDataVec;
  std::vector<PixelData*> warpedCurPixelDataVec;
  std::vector<PixelData*> keyPixelDataVec;


  // KDTree
  KDTree* keyTree2;
  KDTree* keyTree4;


  bool isInit;           // boolean indicating whether it is the first iteration or not.

  // Related to rigid body motion.
  Eigen::MatrixXd tmpXi; // The se(3) from the current keyframe to the current frame.
  Eigen::MatrixXd delXi; // infinitisimal motion update during the optimization.
  Eigen::MatrixXd tmpTransform;
  Eigen::MatrixXd keyTransform;

  std::vector<Eigen::MatrixXd> trajXi; //
  std::vector<Eigen::MatrixXd> trajTransform;

  Eigen::MatrixXd currentPose;

};
#endif
