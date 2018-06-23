#include "edgeicp.h"

#include <iostream>
#include <sys/time.h>


Edgeicp::Edgeicp(Parameters params_) : params(params_) {
  if(params.debug.imgShowFlag == true) {
    cv::namedWindow("current image");
    cv::namedWindow("key image");
    cv::namedWindow("current edge image");
  }
  this->completeFlag = false;
  this->isInit       = true;
  this->numOfImg     = 0;

  this->keyTree2     = NULL;
  this->keyTree4     = NULL;

  this->debugImg     = cv::Mat(240, 320, CV_8UC3);
	this->debugImg     = cv::Scalar(255,255,255);


  // initialize containers
  this->tmpXi        = Eigen::MatrixXd::Zero(6,1);
  this->delXi        = Eigen::MatrixXd::Zero(6,1);
  this->tmpTransform = Eigen::MatrixXd::Zero(4,4);

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

void Edgeicp::delete_pixeldata(std::vector<PixelData*>& pixelDataVec) { // storage deletion.
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
  imgOutputEdge.create(imgInputEdge.size(), CV_8UC1);

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
    const short* imgGradxPtr    = imgGradx.ptr<short>(v);
    const short* imgGradyPtr    = imgGrady.ptr<short>(v);
    const double* imgDepthPtr    = imgDepth.ptr<double>(v);

    for(u = 0; u < imgInputEdge.cols; u++){
      if(*(imgInputEdgePtr++) == 255){
        double invGradNorm = 1.0/sqrt( (double)((*(imgGradxPtr + u))*(*(imgGradxPtr + u)) + (*(imgGradyPtr + u))*(*(imgGradyPtr + u))) );

        Edgeicp::PixelData* tmpPixelData = new Edgeicp::PixelData( (double)u, (double)v, (double)*(imgDepthPtr + u), (double)(*(imgGradxPtr + u))*(double)invGradNorm, (double)(*(imgGradyPtr + u))*(double)invGradNorm );
        //std::cout<<tmpPixelData->u<<std::endl;
        pixelDataVec.push_back(tmpPixelData);
        cnt++;
      }
    }
  }

  std::cout<<"Num of points : "<< cnt <<std::endl;
}

void Edgeicp::downsample_image(const cv::Mat& imgInput, cv::Mat& imgOutput){
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

void Edgeicp::calc_ICP_residual_div(const std::vector<PixelData*>& curPixelDataVec_, const std::vector<PixelData*>& keyPixelDataVec_, const std::vector<int>& rndIdx_, const std::vector<int>& refIdx_, std::vector<double>& residualVec_){

  // input : this->keyPixelDataVec, this->curPixelDataVec, refIdx, rndIdx
  double xr, yr, xc, yc;
  double diff_x, diff_y;
  double grad_x, grad_y;
  double resX, resY, resTotal;
  int rndNum = rndIdx_.size();
  residualVec_.reserve(0); // I assume that the "residualVec_" is given initialized ( empty )

  for(int i = 0; i < rndNum; i++){
    xc = curPixelDataVec_[rndIdx_[i]]->u;
    yc = curPixelDataVec_[rndIdx_[i]]->v;

    xr = keyPixelDataVec_[refIdx_[i]]->u;
    yr = keyPixelDataVec_[refIdx_[i]]->v;

    grad_x = keyPixelDataVec_[refIdx_[i]]->gx;
    grad_y = keyPixelDataVec_[refIdx_[i]]->gy;

    diff_x = xc - xr;
    diff_y = yc - yr;

    resX = diff_x*grad_x;
    resY = diff_y*grad_y;

    resTotal = resX + resY;

    residualVec_.push_back(resTotal);
  }
}
// TODO: !!!
/*
void Edgeicp::calc_ICP_Jacobian_div(){
  double fx = this->params.calib.fx;
  double fy = this->params.calib.fy;
  double cx = this->params.calib.cx;
  double cy = this->params.calib.cy;
  double invZ;

  // X Y Z : warped points.
  for(int i = 0; i < 1; i++){
    invZ = 1.0;
    invZinvZ = invZ*invZ;

    J[0,i] = fx*invZ*gx;
    J[1,i] = fy*invZ*gy;
    J[2,i] = -fx*X*invZinvZ*gx - fy*Y*invZinvZ*gy;
    J[3,i] = -fx*X*Y*invZinvZ*gx - fy*(1+Y*Y*invZinvZ)*gy;
    J[4,i] = fx*(1+X*X*invZinvZ)*gx + fy*X*Y*invZinvZ*gy;
    J[5,i] = -fx*Y*invZ*gx + fy*X*invZ*gy;
  }
}
*/
/* double Edgeicp::update_t_distribution(const std::vector<double>& residualVec_){
  int nSample = 1000;
  int N = residualVec_.size();

  std::vector<int> rndIdx;
  rndIdx.reserve(0);

  std::vector<double> residualVecSampled;
  residualVecSampled.reserve(0);


  if( N >= nSample) {
    rnd::randsample(N, nSample, rndIdx);
    for(int i = 0; i < nSample; i++) residualVecSampled.push_back(residualVec_[rndIdx[i]]);
  }
  else {

  }

  double temp;
  while(1){

    temp = ( nu + 1.0 )/N * summation;
    lambda_curr  = 1.0 / temp;

    if(fabs(lambda_curr - lambda_prev) <= 1e-7 ) break;
    lambda_prev = lambda_curr;
  }

  sigma_new = sqrt(1.0 / )
  return sigma_new;
}*/









/* ======================================================================
 * ============================ RUN FUNCTION ============================
 * ============================ =========================================
 */
void Edgeicp::run() {
  this->completeFlag = true;

  if(this->isInit == true) { // keyframe initialization
    ROS_INFO_STREAM("FIRST ITERATION - keyframe initialize");

    // Initialize point containers
    Edgeicp::delete_pixeldata(this->curPixelDataVec);
    Edgeicp::delete_pixeldata(this->keyPixelDataVec);
    this->tmpXi  = Eigen::MatrixXd::Zero(6,1);
    this->delXi  = Eigen::MatrixXd::Zero(6,1);

    // Initialize the keyImg and keyDepth.
    this->keyImg   = this->curImg;
    this->keyDepth = this->curDepth;

    // Canny edge algorithm to detect the edge pixels.
    Edgeicp::downsample_image(this->keyImg,   this->keyImgLow);
    Edgeicp::downsample_depth(this->keyDepth, this->keyDepthLow);

    // Find gradient
    Edgeicp::calc_gradient(this->keyImgLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, true);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->keyImgLow, this->keyEdgeMap, this->params.canny.lowThres, this->params.canny.highThres);

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
    this->keyTree2 = new KDTree( tmpPixel2Vec, (this->params.hyper.treeDistThres*this->params.hyper.treeDistThres)/(this->params.calib.width*this->params.calib.width));

    // Initialize the current images
    this->keyEdgeMap.copyTo(this->curEdgeMap);
    this->keyEdgeMapValid.copyTo(this->curEdgeMapValid);

    this->keyImgLow.copyTo(this->curImgLow);
    this->keyDepthLow.copyTo(this->curDepthLow);

    this->keyImgGradx.copyTo(this->curImgGradx);
    this->keyImgGrady.copyTo(this->curImgGrady);
    this->keyImgGrad.copyTo(this->curImgGrad);


    //this->curEdgeMap      = this->keyEdgeMap;
    //this->curEdgeMapValid = this->keyEdgeMapValid;

    //this->curImgLow       = this->keyImgLow;
    //this->curDepthLow     = this->keyDepthLow;

    //this->curImgGradx     = this->keyImgGradx;
    //this->curImgGrady     = this->keyImgGrady;
    //this->curImgGrad      = this->keyImgGrad;

    // First keyframe is updated done.
    this->isInit = false;
  }
  else { // After initial images, successively run the algorithm for the current image.
    // initialize containers
    Edgeicp::delete_pixeldata(this->curPixelDataVec);

    // Downsample image ( resolution down upto lvl = 2)
    Edgeicp::downsample_image(this->curImg,   this->curImgLow);
    Edgeicp::downsample_depth(this->curDepth, this->curDepthLow);

    // Find gradient
    Edgeicp::calc_gradient(this->curImgLow, this->curImgGradx, this->curImgGrady, this->curImgGrad, true);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->curImgLow, this->curEdgeMap, this->params.canny.lowThres, this->params.canny.highThres);

    // Find valid edge pixels from depth and gradient test.
    Edgeicp::find_valid_mask(this->curEdgeMap, this->curDepthLow, this->curImgGrad, this->curEdgeMapValid);

    // Extract edge pixels and store in vector.
    Edgeicp::set_edge_pixels(this->curEdgeMapValid, this->curDepthLow, this->curImgGradx, this->curImgGrady, this->curImgGrad, this->curPixelDataVec);

    // Store the pixel coordinates in vector with sampled number of points.
    std::vector<std::vector<double>> tmpPixel2Vec; // For kdtree generation. Temporary vector container.
    tmpPixel2Vec.reserve(0);
    std::vector<int> rndIdx;
    rndIdx.reserve(0);

    rnd::randsample(this->curPixelDataVec.size(), this->params.hyper.nSample, rndIdx); //sampling without replacement

    double invWidth = 1.0 / (double)this->params.calib.width;
    for(int i = 0; i < rndIdx.size(); i++) {
      std::vector<double> tmpPixel2;
      tmpPixel2.push_back(this->curPixelDataVec[rndIdx[i]]->u*invWidth);
      tmpPixel2.push_back(this->curPixelDataVec[rndIdx[i]]->v*invWidth);
      tmpPixel2Vec.push_back(tmpPixel2);
    }

    // ========================== //
    //  Iterative optimization !  //
    // ========================== //
    int    icpIter  = 0;
    double errLast  = 1000000.0, errPrev = 0.0;
    double lambda   = 0.05;
    double stepSize = 0.7;
    std::vector<int> refIdx;

    while(icpIter < this->params.hyper.maxIter) {
      std::vector<double> residualVec;

      if(icpIter < this->params.hyper.shiftIter) { // 4-D kdtree approximated NN search
        this->keyTree2->kdtree_nearest_neighbor(tmpPixel2Vec, refIdx);
      }
      else { // 2-D kdtree search exact NN search
        this->keyTree2->kdtree_nearest_neighbor(tmpPixel2Vec, refIdx);
      }
      // TODO: residual calculation
      Edgeicp::calc_ICP_residual_div(this->curPixelDataVec, this->keyPixelDataVec, rndIdx, refIdx, residualVec);

      for(int k=0; k < residualVec.size(); k++){
        std::cout<< residualVec[k] <<std::endl;
      }

      // TODO: t-distribution update ( update_t_distribution )
      if(icpIter > 5) { // t-distribution weighting after 5 iterations

      }
      // TODO: calculation Jacobian matrix ( calc_ICP_Jacobian_div )

      // TODO: residual reweighting ( update_weight_matrix )

      // TODO: Weighted residual

      // TODO: Hessian calculation

      // TODO: delta_xi calculation.

      // TODO: xi_temp = xi_temp + delta_xi;

      // TODO: iteration stop condition
      if(0) {
        break;
      }

      icpIter++;
      //std::cout<<"      optimization iterations : "<<icpIter<<std::endl;
    }

    // showing the debuging image.
    if(this->params.debug.imgShowFlag == true){
      cv::Scalar colorLine(0,127,255);
      cv::Scalar colorText(120,120,0);
      cv::Scalar colorCircleRef(0,0,0);
      cv::Scalar colorCircleCur(0,0,255);
      double xr, yr, xc, yc;
      for(int i = 0; i < refIdx.size(); i++){
        xr = this->keyPixelDataVec[refIdx[i]]->u;
        yr = this->keyPixelDataVec[refIdx[i]]->v;
        xc = this->curPixelDataVec[rndIdx[i]]->u;
        yc = this->curPixelDataVec[rndIdx[i]]->v;
        cv::line(this->debugImg, cv::Point(xc,yc), cv::Point(xr,yr), colorLine, 2);
        cv::circle(this->debugImg, cv::Point(xr,yr),  1, colorCircleRef, CV_FILLED);
        cv::circle(this->debugImg, cv::Point(xc,yc),  1, colorCircleCur, CV_FILLED);
        //cv::circle(this->curImgLow, cv::Point(xc, yc), 1, colorCircleCur, CV_FILLED);

        putText(this->debugImg, "DEBUG IMAGE", cv::Point(180,180), cv::FONT_HERSHEY_SIMPLEX, 0.7, colorText, 2.0);
      }

      for(int i = 0; i < this->keyPixelDataVec.size(); i++){
        //xr = this->keyPixelDataVec[i]->u;
        //yr = this->keyPixelDataVec[i]->v;
        //cv::circle(this->keyImgLow, cv::Point(xr, yr), 1, colorCircleRef, CV_FILLED);
      }

      this->curEdgeMap.copyTo(this->debugEdgeImg);
    }



    // If the distance from the current keyframe to the current frame exceeds the threshold, renew the key frame
    if(this->numOfImg % 10 == 1){
      this->tmpXi       = Eigen::MatrixXd::Zero(6,1);
      this->delXi       = Eigen::MatrixXd::Zero(6,1);

      // free dynamic allocations
      Edgeicp::delete_pixeldata(this->curPixelDataVec);
      Edgeicp::delete_pixeldata(this->keyPixelDataVec);

      delete this->keyTree2;
      delete this->keyTree4;

      // reset the keyImg and keyDepth.
      this->curEdgeMap.copyTo(this->keyEdgeMap);
      this->curEdgeMapValid.copyTo(this->keyEdgeMapValid);

      this->curImgLow.copyTo(this->keyImgLow);
      this->curDepthLow.copyTo(this->keyDepthLow);

      this->curEdgeMap.copyTo(this->keyEdgeMap);

      this->curImgGradx.copyTo(this->keyImgGradx);
      this->curImgGrady.copyTo(this->keyImgGrady);
      this->curImgGrad.copyTo(this->keyImgGrad);


      // Canny edge algorithm to detect the edge pixels.
      cv::Canny(this->keyImgLow, this->keyEdgeMap, this->params.canny.lowThres, this->params.canny.highThres);

      // Find valid edge pixels from depth and gradient test.
      Edgeicp::find_valid_mask(this->keyEdgeMap, this->keyDepthLow, this->keyImgGrad, this->keyEdgeMapValid);

      // Extract edge pixels and store in vector.
      Edgeicp::set_edge_pixels(this->keyEdgeMapValid, this->keyDepthLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, this->keyPixelDataVec);

      // At this location, kd tree re-construction.
      double invWidth = 1.0 / (double)this->params.calib.width;
      std::vector<std::vector<double>> tmpPixel2Vec;
      tmpPixel2Vec.reserve(0);

      for(int i = 0; i < this->keyPixelDataVec.size(); i++) {
      	std::vector<double> tmpPixel2;
      	tmpPixel2.push_back(this->keyPixelDataVec[i]->u*invWidth);
        tmpPixel2.push_back(this->keyPixelDataVec[i]->v*invWidth);
      	tmpPixel2Vec.push_back(tmpPixel2);
      }


      // Build new k-d tree.
      this->keyTree2 = new KDTree( tmpPixel2Vec, (this->params.hyper.treeDistThres*this->params.hyper.treeDistThres)/(this->params.calib.width*this->params.calib.width));
    }
  }


  if(this->params.debug.imgShowFlag == true) {
    cv::Mat scaledImg;
    double min, max;
    cv::minMaxIdx(this->debugEdgeImg, &min, &max);
    cv::convertScaleAbs(this->debugEdgeImg, scaledImg, 255 / max);

    cv::imshow("current image", this->curImgLow);
    cv::imshow("key image", this->keyImgLow);
    cv::imshow("current edge image", this->debugImg);
    //cv::imshow("current edge image", scaledImg);
    cv::waitKey(3);
    this->debugImg   =  cv::Scalar(255,255,255);
  }
}
