#include "edgeicp.h"

#include <iostream>
#include <sys/time.h>


Edgeicp::Edgeicp(Parameters params_) : params(params_) {
  if(params.debug.imgShowFlag == true) cv::namedWindow("current image");
  this->completeFlag = false;
  this->isInit       = true;
  this->numOfImg     = 0;

  this->keyTree2     = NULL;
  this->keyTree4     = NULL;
}

Edgeicp::~Edgeicp() {
  // All dynamic allocations must be deleted !!
  Edgeicp::delete_pixeldata(this->curPixelDataVec);
  Edgeicp::delete_pixeldata(this->keyPixelDataVec);

  // Tree collapse
  if(this->keyTree2 != NULL){
    delete this->keyTree2;
    std::cout<<"Tree2 is deleted !"<<std::endl;
  }
  if(this->keyTree4 != NULL){
    delete this->keyTree4;
    std::cout<<"Tree4 is deleted !"<<std::endl;
  }

  ROS_INFO_STREAM("Edgeicp node is terminated.\n");
}

void Edgeicp::image_acquisition(const cv::Mat& img_, const cv::Mat& depth_, const TopicTime& curTime_){
  this->curImg    = img_.clone();
  this->curDepth  = depth_.clone();
  this->curTime   = curTime_;
  std::cout<<"# of images : "<< ++this->numOfImg <<std::endl;
}

/*Edgeicp::PixelData* Edgeicp::new_pixeldata(double* u_, double* v_, double* d_, double* gx_, double* gy_){
  Edgeicp::PixelData* pixelData;
  if( (pixelData = (Edgeicp::PixelData*)malloc(sizeof(Edgeicp::PixelData)) ) == NULL){
    printf("error allocating new pixeldata ! \n");
    exit(1);
  }
  pixelData->u = u_;
  pixelData->v = v_;
  pixelData->d = d_;
  pixelData->gx = gx_;
  pixelData->gy = gy_;

  return pixelData;
}*/

void Edgeicp::delete_pixeldata(std::vector<PixelData*>& pixelDataVec) {
  int len = pixelDataVec.size();
  if(len > 0){
    for(std::vector<PixelData*>::iterator it = pixelDataVec.begin(); it != pixelDataVec.end(); it++) {
      delete (*it);
    }
    pixelDataVec.clear();
  }
  else printf("No edge pixels to be deleted !\n");
}

void Edgeicp::find_valid_mask(const cv::Mat& imgInputEdge, const cv::Mat& imgDepth, const cv::Mat& imgGrad, cv::Mat& imgOutputEdge){
  int numValidPixels = 0; // initialize
  imgOutputEdge.create(imgInputEdge.size(), CV_8U);

  int u, v;
  for(v = 0; v < imgInputEdge.rows; v++){
    const uchar  *imgInputEdgePtr = imgInputEdge.ptr<uchar>(v);
    const ushort *imgDepthPtr     = imgDepth.ptr<ushort>(v);
    uchar *imgOutputEdgePtr       = imgOutputEdge.ptr<uchar>(v);

    for(u = 0; u < imgInputEdge.cols; u++){
      if(*(imgInputEdgePtr++) > 0 & *(imgDepthPtr++) > 0){
        *(imgOutputEdgePtr++) = 255;
        numValidPixels++;
      }
      else{
        *(imgOutputEdgePtr++) = 0;
      }
    }
  }
}


void Edgeicp::set_edge_pixels(const cv::Mat& imgInputEdge, const cv::Mat& imgDepth, const cv::Mat& imgGradx, const cv::Mat& imgGrady, const cv::Mat& imgGrad, std::vector<Edgeicp::PixelData*>& pixelDataVec){
  int cnt = 0;
  int u, v;
  pixelDataVec.reserve(0); // initialize the pixelDataVec vector.

  for(v = 0; v < imgInputEdge.rows; v++){
    const uchar* imgInputEdgePtr = imgInputEdge.ptr<uchar>(v);
    const double* imgGradxPtr    = imgGradx.ptr<double>(v);
    const double* imgGradyPtr    = imgGrady.ptr<double>(v);
    const double* imgDepthPtr    = imgDepth.ptr<double>(v);

    for(u = 0; u < imgInputEdge.cols; u++){
      if(*(imgInputEdgePtr++) == 255){
        Edgeicp::PixelData* tmpPixelData = new Edgeicp::PixelData( (double)u, (double)v, *(imgDepthPtr + u), *(imgGradxPtr + v), *(imgGradyPtr + u) );
        pixelDataVec.push_back(tmpPixelData);
        cnt++;
      }
    }
  }

  std::cout<<"Num of points : "<<cnt<<std::endl;
}

void Edgeicp::downsample_iamge(const cv::Mat& imgInput, cv::Mat& imgOutput){
  imgOutput.create(cv::Size(imgInput.cols / 2, imgInput.rows / 2), imgInput.type());
  if(imgInput.type() != 0){
    std::cout<<"Gray image type is not a CV_8UC1 (uchar) ! "<<std::endl;
    exit(1);
  }
  int u2, u21;
  int u,  v;

  for(v = 0; v < imgOutput.rows; ++v) {
    const uchar* imgInputRowPtr1 = imgInput.ptr<uchar>(2*v);
    const uchar* imgInputRowPtr2 = imgInput.ptr<uchar>(2*v + 1);
    uchar* imgOutputRowPtr = imgOutput.ptr<uchar>(v);
      for(u = 0; u < imgOutput.cols; ++u) {
      u2  = 2*u;
      u21 = u2 + 1;
      imgOutputRowPtr[u] =  (uchar)( ( imgInputRowPtr1[u2] + imgInputRowPtr1[u21] + imgInputRowPtr2[u2] + imgInputRowPtr2[u21] ) / 4.0 );
    }
  }
}


void Edgeicp::downsample_depth(const cv::Mat& imgInput, cv::Mat& imgOutput) {
    imgOutput.create(cv::Size(imgInput.size().width / 2, imgInput.size().height / 2), imgInput.type());
    if(imgInput.type() != 2){
      std::cout<<"Depth image type is not a CV_16UC1 (ushort) ! "<<std::endl;
      exit(1);
    }
    int u0 = 0, u1 = 0, v0 = 0, v1 = 0;
    int v, u ;
    ushort sum, cnt;
    for(v = 0; v < imgOutput.rows; ++v) {
        for(u = 0; u < imgOutput.cols; ++u) {
            u0 = u * 2;
            u1 = u0 + 1;
            v0 = v * 2;
            v1 = v0 + 1;

            // initialize
            sum = 0;
            cnt = 0;

            if((imgInput.at<ushort>(v0, u0) > 0.01f)) {
                sum += imgInput.at<ushort>(v0, u0);
                cnt += 1;
            }
            if((imgInput.at<ushort>(v0, u1) > 0.01f)) {
                sum += imgInput.at<ushort>(v0, u1);
                cnt += 1;
            }
            if((imgInput.at<ushort>(v1, u0) > 0.01f)) {
                sum += imgInput.at<ushort>(v1, u0);
                cnt += 1;
            }
            if((imgInput.at<ushort>(v1, u1) > 0.01f)) {
                sum += imgInput.at<ushort>(v1, u1);
                cnt += 1;
            }
            if(cnt > 0) imgOutput.at<ushort>(v, u) = ( sum / cnt );
            else imgOutput.at<ushort>(v, u) = 0;
        }
    }
}


void Edgeicp::calc_gradient(const cv::Mat& imgInput, cv::Mat& imgGradx, cv::Mat& imgGrady, cv::Mat& imgGrad, const bool& doGaussian){
  // calculate gradient along each direction.
  cv::Sobel(imgInput, imgGradx, CV_16S, 1,0,3,1,0,cv::BORDER_DEFAULT); // CV_16S : short -32768~32768, CV_64F : double
  cv::Sobel(imgInput, imgGrady, CV_16S, 0,1,3,1,0,cv::BORDER_DEFAULT);

  if(doGaussian == true){ // apply Gaussian filtering or not
    cv::GaussianBlur(imgGradx, imgGradx, cv::Size(3,3),0,0);
    cv::GaussianBlur(imgGrady, imgGrady, cv::Size(3,3),0,0);
  }

  // calculate gradient norm
  imgGrad.create(imgInput.size(), CV_64F);
  int u, v;
  for(v = 0; v < imgInput.rows; v++){
    short* imgGradxPtr = imgGradx.ptr<short>(v);
    short* imgGradyPtr = imgGrady.ptr<short>(v);
    double* imgGradPtr = imgGrad.ptr<double>(v);
    for(u = 0; u < imgInput.cols; u++){
      *(imgGradPtr+u) = sqrt( (double)( ( *(imgGradxPtr + u) ) * ( *(imgGradxPtr + u) ) + ( *(imgGradyPtr + u) ) * ( *(imgGradyPtr + u) ) ) );
    }
  }
}

void Edgeicp::getMotion(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw){
}






/* ======================================================================
 * ============================ RUN FUNCTION ============================
 * ============================ =========================================
 */
void Edgeicp::run() {
  this->completeFlag = true; //

  if(this->isInit == true) { // keyframe initialization
    ROS_INFO_STREAM("FIRST ITERATION - keyframe initialize");
    this->tmpXi  = Eigen::MatrixXd::Zero(6,1);
    this->delXi  = Eigen::MatrixXd::Zero(6,1);

    // Initialize the keyImg and keyDepth.
    this->keyImg   = this->curImg;
    this->keyDepth = this->curDepth;

    // Canny edge algorithm to detect the edge pixels.
    Edgeicp::downsample_iamge(this->keyImg,   this->keyImgLow);
    Edgeicp::downsample_depth(this->keyDepth, this->keyDepthLow);

    // Find gradient
    Edgeicp::calc_gradient(this->keyImgLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, false);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->keyImgLow, this->keyEdgeMap, 10, 100);

    // Find valid edge pixels from depth and gradient test.
    Edgeicp::find_valid_mask(this->keyEdgeMap, this->keyDepthLow, this->keyImgGrad, this->keyEdgeMapValid);

    // Extract edge pixels and store in vector.
    Edgeicp::set_edge_pixels(this->keyEdgeMapValid, this->keyDepthLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, this->keyPixelDataVec);

    // At this location, kd tree construction.
    double invWidth = 1.0 / (double)this->params.calib.width;
    std::vector<std::vector<double>> tmpPixel2Vec;
    tmpPixel2Vec.reserve(0);
    for(int i = 0; i < this->keyPixelDataVec.size(); i++) {
    	std::vector<double> tmpPixel2;
    	tmpPixel2.push_back(this->keyPixelDataVec[i]->u*invWidth);
      tmpPixel2.push_back(this->keyPixelDataVec[i]->v*invWidth);
    	tmpPixel2Vec.push_back(tmpPixel2);
    }
    keyTree2 = new KDTree( tmpPixel2Vec, (this->params.hyper.treeDistThres*this->params.hyper.treeDistThres)/(this->params.calib.width*this->params.calib.width));

    // Initialize the current images
    this->curEdgeMap      = this->keyEdgeMap;
    this->curEdgeMapValid = this->keyEdgeMapValid;

    this->curImgLow       = this->keyImgLow;
    this->curDepthLow     = this->keyDepthLow;

    this->curImgGradx     = this->keyImgGradx;
    this->curImgGrady     = this->keyImgGrady;
    this->curImgGrad      = this->keyImgGrad;

    // First keyframe is updated done.
    this->isInit = false;
  }
  else { // After initial images, successively run the algorithm for the current image.
    // Downsample image ( resolution down upto lvl = 2)
    Edgeicp::downsample_iamge(this->curImg, this->curImgLow);
    Edgeicp::downsample_depth(this->curDepth, this->curDepthLow);

    // Find gradient
    Edgeicp::calc_gradient(this->curImgLow, this->curImgGradx, this->curImgGrady, this->curImgGrad, false);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->curImgLow, this->curEdgeMap, 120, 200);

    // Find valid edge pixels from depth and gradient test.
    Edgeicp::find_valid_mask(this->curEdgeMap, this->curDepthLow, this->curImgGrad, this->curEdgeMapValid);

    // Extract edge pixels and store in vector.
    Edgeicp::set_edge_pixels(this->curEdgeMapValid, this->curDepthLow, this->curImgGradx, this->curImgGrady, this->curImgGrad, curPixelDataVec);

    // Store the pixel coordinates in vector.
    double invWidth = 1.0 / (double)this->params.calib.width;
    std::vector<std::vector<double>> tmpPixel2Vec;
    tmpPixel2Vec.reserve(0);
    for(int i = 0; i < this->curPixelDataVec.size(); i++) {
      std::vector<double> tmpPixel2;
      tmpPixel2.push_back(this->curPixelDataVec[i]->u*invWidth);
      tmpPixel2.push_back(this->curPixelDataVec[i]->v*invWidth);
      tmpPixel2Vec.push_back(tmpPixel2);
    }

    // ========================== //
    //  Iterative optimization !  //
    // ========================== //
    int    icpIter  = 0;
    double errLast  = 1000000, errPrev;
    double lambda   = 0.05;
    double stepSize = 0.7;
    while(icpIter < this->params.hyper.maxIter){
      std::vector<int> refIdx;

      if(icpIter < this->params.hyper.shiftIter) { // 4-D kdtree approximated NN search
        keyTree2->kdtree_nearest_neighbor(tmpPixel2Vec, refIdx);
      }
      else { // 2-D kdtree search exact NN search

      }

      if(icpIter > 5){ // t-distribution weighting after 5 iterations

      }

      if(0){ // iteration stop

        break;
      }
      icpIter++;
      //std::cout<<"      optimization iterations : "<<icpIter<<std::endl;
    }

    if(0){ // if the distance from the current keyframe to the current frame exceeds the threshold, renew the key frame
      this->tmpXi       = Eigen::MatrixXd::Zero(6,1);
      this->delXi       = Eigen::MatrixXd::Zero(6,1);

      this->keyImg      = this->curImg;
      this->keyDepth    = this->curDepth;

      this->keyImgLow   = this->curImgLow;
      this->keyDepthLow = this->curDepthLow;

      this->keyEdgeMap  = this->curEdgeMap;

      Edgeicp::delete_pixeldata(this->keyPixelDataVec);
    }





    // free dynamic allocations
    Edgeicp::delete_pixeldata(this->curPixelDataVec);
  }


  if(this->params.debug.imgShowFlag == true){
    cv::Mat scaledImg;
    double min, max;
    cv::minMaxIdx(this->curDepthLow, &min, &max);
    cv::convertScaleAbs(this->curDepthLow, scaledImg, 255 / max);
    cv::imshow("current image", this->curEdgeMap);
    cv::waitKey(3);
  }
}
