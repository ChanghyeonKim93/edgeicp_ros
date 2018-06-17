#include <ros/ros.h>
#include "edgeicp.h"

#include <iostream>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// These headers are needed to subscribe stereo image pair.
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#define APPROXIMATE 1

#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif

bool imgUpdated    = false;
bool algorithmDone = true;

typedef std::string TopicTime;

TopicTime imgTime;

cv::Mat curGrayImg, curDepth;

inline std::string dtos(double x);
void image_callback(const sensor_msgs::ImageConstPtr& msgColor , const sensor_msgs::ImageConstPtr& msgDepth);

int main(int argc, char **argv) {

	ros::init(argc, argv, "edgeicp_node");
	ros::NodeHandle nh("~");

	std::string imgTopicName, depthTopicName;
	bool dbgFlag;

	// Get ROS parameters from launch file.
	ros::param::get("~color_topic_name", imgTopicName);
	ros::param::get("~depth_topic_name", depthTopicName);
	ros::param::get("~debug_flag",dbgFlag);

	// Initialize subscribers.
	message_filters::Subscriber<sensor_msgs::Image> colorImgSubs(nh , imgTopicName , 1 );
	message_filters::Subscriber<sensor_msgs::Image> depthImgSubs(nh , depthTopicName , 1 );

	#ifdef EXACT
	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
	#endif
	#ifdef APPROXIMATE
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
	#endif

	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), colorImgSubs, depthImgSubs );
	sync.registerCallback(boost::bind(&image_callback, _1, _2));

	// =======================================================
	// =======================================================
	// =================== ALGORITHM PART ====================
	// =======================================================
	// =======================================================

	// Define algorithm parameters
	Edgeicp::Parameters params;
	params.debug.imgShowFlag   = dbgFlag;

	params.calib.fx 					 = 620.608832234754;
	params.calib.fy            = 619.113993685335;
	params.calib.cx            = 323.902900972212;
	params.calib.cy            = 212.418428046497;
	params.calib.depthScale    = 1000.0;
	params.calib.width         = 640;
	params.calib.height      	 = 480;

	params.hyper.nSample       = 600;  // the number of sub sampling method.
	params.hyper.maxIter       = 3;   // maximum iteration number of optimization. ( 20 ~ 30 )
	params.hyper.shiftIter     = 7;    // find correspondences by 4 dimensions until shiftIterations. After, we use 2 dimensions matcher.
	params.hyper.treeDistThres = 15.0; // distance thresholding during kd tree searching.
	params.hyper.transThres    = 0.05; //
	params.hyper.rotThres      = 3.0;

	params.canny.lowThres			 = 30;
	params.canny.highThres		 = 120;


	Edgeicp *edgeicp = new Edgeicp(params);

	// ROS spinning.
	while(ros::ok()) {
		ros::spinOnce(); // VERY FAST, consumes negligibly small time !!!
		if(imgUpdated == true) {
			imgUpdated = false;

			edgeicp->image_acquisition(curGrayImg, curDepth, imgTime);
			edgeicp->run();
		}
	}

	// Cease the code.

	delete edgeicp;
	ROS_INFO_STREAM("CEASE - edgeicp ");
	return 0;
}





/* ======================================================================= *
 * ======================================================================= *
 * =========================== implementations =========================== *
 * ======================================================================= *
 * ======================================================================= */

 inline std::string dtos(double x){
 	std::stringstream s;
 	s<<std::setprecision(6) << std::fixed << x;
 	return s.str();
 }


 void image_callback(const sensor_msgs::ImageConstPtr& msgColor , const sensor_msgs::ImageConstPtr& msgDepth){
 	cv_bridge::CvImagePtr imgColor, imgDepth;

 	try{
 		imgColor = cv_bridge::toCvCopy(*msgColor,  sensor_msgs::image_encodings::BGR8);
 	}
 	catch (cv_bridge::Exception& e) 	{
 		ROS_ERROR("cv_bridge exception:  %s", e.what());
 		return;
 	}
 	try{
 		// imgDepth = cv_bridge::toCvCopy(*msgDepth, sensor_msgs::image_encodings::TYPE_16UC1); // for Intel Realsense R200
	  imgDepth = cv_bridge::toCvCopy(*msgDepth, sensor_msgs::image_encodings::TYPE_32FC1); // for ASUS xtion pro live
 	}
 	catch (cv_bridge::Exception& e) 	{
 		ROS_ERROR("cv_bridge exception:  %s", e.what());
 		return;
 	}

 	cv::Mat& matColorImg  = imgColor->image;
 	//cv::Mat& matDepth     = imgDepth->image;

 	// current gray image ( global variable ).
 	cv::cvtColor(matColorImg, curGrayImg, CV_RGB2GRAY);
 	//curDepth = matDepth;
	curDepth = cv::Mat::ones(480, 640, CV_16UC1)*244;
 	double curTime_tmp = (double)(msgColor->header.stamp.sec*1e6+msgColor->header.stamp.nsec/1000)/1000000.0;
 	imgTime = dtos(curTime_tmp);
 	imgUpdated = true;
 	ROS_INFO_STREAM("Image subsc - RGBD images are updated.");
 }
