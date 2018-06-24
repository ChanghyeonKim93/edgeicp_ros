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
  Edgeicp::delete_pixeldata(this->warpedCurPixelDataVec);

  // Tree collapse
  if(this->keyTree2 != NULL){
    delete this->keyTree2;
    std::cout<<"Tree(2D) is deleted !"<<std::endl;
  }
  if(this->keyTree4 != NULL){
    delete this->keyTree4;
    std::cout<<"Tree(4D) is deleted !"<<std::endl;
  }

  ROS_INFO_STREAM("Edgeicp node is terminated.\n");
}

void Edgeicp::image_acquisition(const cv::Mat& img_, const cv::Mat& depth_, const TopicTime& curTime_){
  this->curImg    = img_.clone();
  this->curDepth  = depth_.clone();
  Edgeicp::validify_depth(this->curDepth);

  this->curTime   = curTime_;
  std::cout<<"# of images : "<< (++this->numOfImg) <<std::endl;
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
  int numAllPixels   = 0;
  imgOutputEdge.create(imgInputEdge.size(), CV_8UC1);

  int u, v;
  for(v = 0; v < imgInputEdge.rows; v++)
  {
    const uchar  *imgInputEdgePtr = imgInputEdge.ptr<uchar>(v);
    const double *imgDepthPtr     = imgDepth.ptr<double>(v);
    uchar *imgOutputEdgePtr       = imgOutputEdge.ptr<uchar>(v);

    for(u = 0; u < imgInputEdge.cols; u++)
    {
      if(*(imgInputEdgePtr++) > 0 & *(imgDepthPtr++) > 0)
      {
        *(imgOutputEdgePtr++) = 255;
        numValidPixels++;
      }
      else
      {
        *(imgOutputEdgePtr++) = 0;
      }
      if(*imgInputEdgePtr > 0) numAllPixels++;
    }
  }
}


void Edgeicp::set_edge_pixels(const cv::Mat& imgInputEdge, const cv::Mat& imgDepth, const cv::Mat& imgGradx, const cv::Mat& imgGrady, const cv::Mat& imgGrad, std::vector<Edgeicp::PixelData*>& pixelDataVec){
  int cnt = 0;
  int u, v;
  pixelDataVec.reserve(0); // initialize the pixelDataVec vector.

  for(v = 0; v < imgInputEdge.rows; v++)
  {
    const uchar* imgInputEdgePtr = imgInputEdge.ptr<uchar>(v);
    const short* imgGradxPtr     = imgGradx.ptr<short>(v);
    const short* imgGradyPtr     = imgGrady.ptr<short>(v);
    const double* imgDepthPtr    = imgDepth.ptr<double>(v);

    for(u = 0; u < imgInputEdge.cols; u++)
    {
      if(*(imgInputEdgePtr++) == 255)
      {
        double invGradNorm = 1.0/sqrt( (double)((*(imgGradxPtr + u))*(*(imgGradxPtr + u)) + (*(imgGradyPtr + u))*(*(imgGradyPtr + u))) );
        double X_, Y_, Z_, u_, v_, d_, gx_, gy_;
        u_ = (double)u;
        v_ = (double)v;
        d_ = (double)(*(imgDepthPtr + u));
        X_ = (u_ - this->params.calib.cx)/this->params.calib.fx*d_;
        Y_ = (v_ - this->params.calib.cy)/this->params.calib.fy*d_;
        Z_ = d_;
        gx_= (double)(*(imgGradxPtr + u))*(double)invGradNorm;
        gy_= (double)(*(imgGradyPtr + u))*(double)invGradNorm;

        Edgeicp::PixelData* tmpPixelData = new Edgeicp::PixelData(u_, v_, d_, X_, Y_, Z_, gx_, gy_);
        pixelDataVec.push_back(tmpPixelData);
        cnt++;
      }
    }
  }

  std::cout<<"Num of points : "<< cnt <<std::endl;
}

void Edgeicp::downsample_image(const cv::Mat& imgInput, cv::Mat& imgOutput) {
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
    if(imgInput.type() != 6)
    {
      std::cout<<"Depth image type is not a CV_64FC1 (double) ! "<<std::endl;
      exit(1);
    }
    imgOutput.create(cv::Size(imgInput.size().width / 2, imgInput.size().height / 2), imgInput.type());
    int u0 = 0, u1 = 0, v0 = 0, v1 = 0;
    int v, u ;
    double sum, cnt;
    for(v = 0; v < imgOutput.rows; ++v) {
        for(u = 0; u < imgOutput.cols; ++u) {
            u0 = u * 2;
            u1 = u0 + 1;
            v0 = v * 2;
            v1 = v0 + 1;

            // initialize
            sum = 0;
            cnt = 0;
            if( isnan(imgInput.at<double>(v0, u0)) == 0 && (imgInput.at<double>(v0, u0) > 0.01)) {
                sum += imgInput.at<double>(v0, u0);
                cnt += 1;
            }
            if(!isnan(imgInput.at<double>(v0, u1)) == 0 &&(imgInput.at<double>(v0, u1) > 0.01)) {
                sum += imgInput.at<double>(v0, u1);
                cnt += 1;
            }
            if(!isnan(imgInput.at<double>(v1, u0)) == 0 &&(imgInput.at<double>(v1, u0) > 0.01)) {
                sum += imgInput.at<double>(v1, u0);
                cnt += 1;
            }
            if(!isnan(imgInput.at<double>(v1, u1)) == 0 &&(imgInput.at<double>(v1, u1) > 0.01)) {
                sum += imgInput.at<double>(v1, u1);
                cnt += 1;
            }
            if(isnan(sum) || isnan(cnt)){
              std::cout<<"NaN depth ERROR."<<std::endl;
              exit(1);
            }
            if(cnt > 0) imgOutput.at<double>(v, u) = ( sum / cnt );
            else imgOutput.at<double>(v, u) = 0;
        }
    }
}

void Edgeicp::validify_depth(cv::Mat& imgInput) {
    if(imgInput.type() != 6)
    {
      std::cout<<"Depth image type is not a CV_64FC1 (double) ! "<<std::endl;
      exit(1);
    }
    for(int i = 0; i < imgInput.rows; i++)
    {
      double* imgInputPtr = imgInput.ptr<double>(i);
      for(int u =0; u<imgInput.cols; u++)
      {
        if(std::isnan(*(imgInputPtr+u))) exit(1);
        if(std::isnan(*(imgInputPtr+u))) *(imgInputPtr+u)=0.0;
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

void Edgeicp::print_motion(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw){
  std::cout<<"Current position - x:"<<x<<", y:"<<y<<", z:"<<z<<", roll:"<<roll<<", pitch:"<<pitch<<", yaw"<<yaw<<std::endl;
}

void Edgeicp::calc_icp_residual_div(const std::vector<PixelData*>& curPixelDataVec_, const std::vector<PixelData*>& keyPixelDataVec_, const std::vector<int>& rndIdx_, const std::vector<int>& refIdx_, Eigen::MatrixXd& residual_){

  // input : this->keyPixelDataVec, this->curPixelDataVec, refIdx, rndIdx
  double xr, yr, xc, yc;
  double diff_x, diff_y;
  double grad_x, grad_y;
  double resX, resY, resTotal;
  int rndNum = rndIdx_.size();
  //residualVec_.reserve(0); // I assume that the "residualVec_" is given initialized ( empty )

  for(int i = 0; i < rndNum; i++)
  {
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

    residual_(i,0) = resTotal;
    //residualVec_.push_back(resTotal);
  }
}

void Edgeicp::initialize_pixeldata(std::vector<PixelData*>& inputPixelDataVec_, const int& len_){ // fill all entry 0.
  // Firstly, delete all entry in the std::vector<PixelData*>.
  Edgeicp::delete_pixeldata(inputPixelDataVec_);

  // Insert new PixelData* pointers which are filled with .
  for(int k = 0; k < len_; k++) {
    Edgeicp::PixelData* tmpPixelData = new Edgeicp::PixelData( -1, -1, -1, 0, 0, 0, -1, -1);
    inputPixelDataVec_.push_back(tmpPixelData);
  }
};


void Edgeicp::warp_pixel_points(const std::vector<PixelData*> inputPixelDataVec_, const Eigen::MatrixXd& tmpXi_, std::vector<PixelData*>& warpedPixelDataVec_){
  Eigen::MatrixXd gTmp             = Eigen::MatrixXd::Zero(4,4); // SE(3) (rigid body motion)
  Eigen::MatrixXd originalPointTmp = Eigen::MatrixXd::Zero(4,1); // Homogeneous coordinate.
  Eigen::MatrixXd warpedPointTmp   = Eigen::MatrixXd::Zero(4,1); // Homogeneous coordinate.
  lie::se3_exp(tmpXi_, gTmp); // calculate SE3 from se3 tmpXi_.

  double uTmp, vTmp, dTmp, XTmp, YTmp, ZTmp;

  for(int k = 0; k < inputPixelDataVec_.size(); k++)
  {
    uTmp = inputPixelDataVec_[k]->u;
    vTmp = inputPixelDataVec_[k]->v;
    dTmp = inputPixelDataVec_[k]->d;
    XTmp = inputPixelDataVec_[k]->X;
    YTmp = inputPixelDataVec_[k]->Y;
    ZTmp = inputPixelDataVec_[k]->Z;

    if(isnan(inputPixelDataVec_[k]->d)){
      std::cout<<"NaN error - function[warp_pixel_points]"<<std::endl;
      exit(1);
    }
    if(inputPixelDataVec_[k]->d <= 0){
      std::cout<<"Zero error - function[warp_pixel_points]"<<std::endl;
      exit(1);
    }

    originalPointTmp(0,0) = XTmp; // (u-cu)/fu*depth, unit : [meter]
    originalPointTmp(1,0) = YTmp; // (v-cv)/fv*depth, unit : [meter]
    originalPointTmp(2,0) = ZTmp; // depth [meter]
    originalPointTmp(3,0) = 1.0;

    // warp points.
    warpedPointTmp = gTmp * originalPointTmp;

    // project point onto pixel plane and
    // allocate the point information into warpedPixelDataVec

    double invD = 1.0/warpedPointTmp(2,0);
    warpedPixelDataVec_[k]->u = this->params.calib.fx*warpedPointTmp(0,0)*invD + this->params.calib.cx;
    warpedPixelDataVec_[k]->v = this->params.calib.fy*warpedPointTmp(1,0)*invD + this->params.calib.cy;
    warpedPixelDataVec_[k]->d = warpedPointTmp(2,0);

    warpedPixelDataVec_[k]->X = warpedPointTmp(0,0);
    warpedPixelDataVec_[k]->Y = warpedPointTmp(1,0);
    warpedPixelDataVec_[k]->Z = warpedPointTmp(2,0);

  }
};

void Edgeicp::convert_pixeldatavec_to_vecvec2d(const std::vector<PixelData*>& pixelDataVec_, const std::vector<int>& indVec_, std::vector<std::vector<double>>& tmpPixel2Vec_){

      if(tmpPixel2Vec_.size() > 0) tmpPixel2Vec_.reserve(0);
      tmpPixel2Vec_.reserve(0);

      double invWidth = 1.0 / (double)this->params.calib.width;
      for(int i = 0; i < indVec_.size(); i++)
      {
        std::vector<double> tmpPixel2;
        tmpPixel2.push_back( pixelDataVec_[indVec_[i]]->u*invWidth);
        tmpPixel2.push_back( pixelDataVec_[indVec_[i]]->v*invWidth);
        tmpPixel2Vec_.push_back(tmpPixel2);
      }
};

// TODO: !!!

void Edgeicp::calc_ICP_Jacobian_div(const std::vector<PixelData*>& warpedCurPixelDataVec_, const std::vector<PixelData*>& keyPixelDataVec_, const std::vector<int>& rndIdx_, const std::vector<int>& refIdx_, Eigen::MatrixXd& J_) {
  double fx = this->params.calib.fx;
  double fy = this->params.calib.fy;
  double X, Y, Z, gx, gy, invZ, invZinvZ, fxgx, fygy;

  // X Y Z : warped points.
  for(int i = 0; i < rndIdx_.size(); i++)
  {
    X = warpedCurPixelDataVec_[rndIdx_[i]]->X;
    Y = warpedCurPixelDataVec_[rndIdx_[i]]->Y;
    Z = warpedCurPixelDataVec_[rndIdx_[i]]->Z;
    gx = keyPixelDataVec_[refIdx_[i]]->gx;
    gy = keyPixelDataVec_[refIdx_[i]]->gy;
    fxgx = fx*gx;
    fygy = fy*gy;
    invZ = 1.0/Z;
    invZinvZ = invZ*invZ;
    /*
    J_(i,0) = fx*invZ*gx; // 2
    J_(i,1) = fy*invZ*gy; // 2
    J_(i,2) = -fx*X*invZinvZ*gx - fy*Y*invZinvZ*gy; // 6
    J_(i,3) = -fx*X*Y*invZinvZ*gx - fy*(1+Y*Y*invZinvZ)*gy; // 8
    J_(i,4) = fx*(1+X*X*invZinvZ)*gx + fy*X*Y*invZinvZ*gy; // 8
    J_(i,5) = -fx*Y*invZ*gx + fy*X*invZ*gy; // 6 , total 32
    */
    J_(i,0) = fxgx*invZ; // 1
    J_(i,1) = fygy*invZ; // 1
    J_(i,2) = -fxgx*X*invZinvZ - fygy*Y*invZinvZ; // 4
    J_(i,3) = -fxgx*X*Y*invZinvZ - fygy*(1+Y*Y*invZinvZ); // 6
    J_(i,4) = fxgx*(1+X*X*invZinvZ) + fygy*X*Y*invZinvZ; // 6
    J_(i,5) = -fxgx*Y*invZ + fygy*X*invZ; // 4 , total 22
  }
}


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

  if(this->isInit == true) // keyframe initialization
  {
    ROS_INFO_STREAM("FIRST ITERATION - keyframe initialize");

    this->tmpXi  = Eigen::MatrixXd::Zero(6,1);
    this->delXi  = Eigen::MatrixXd::Zero(6,1);

    // Initialize the keyImg and keyDepth.
    this->curImg.copyTo(this->keyImg);
    this->curDepth.copyTo(this->keyDepth);

    // Canny edge algorithm to detect the edge pixels.
    Edgeicp::downsample_image(this->keyImg,   this->keyImgLow);
    Edgeicp::downsample_depth(this->keyDepth, this->keyDepthLow);

    // Find gradient
    Edgeicp::calc_gradient(this->keyImgLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, true);

    // Canny edge algorithm to detect the edge pixels.
    cv::Canny(this->keyImgLow, this->keyEdgeMap, this->params.canny.lowThres, this->params.canny.highThres);

    // Find  edge pixels from depth and gradient test.
    Edgeicp::find_valid_mask(this->keyEdgeMap, this->keyDepthLow, this->keyImgGrad, this->keyEdgeMapValid);

    // Extract edge pixels and store in vector.
    Edgeicp::set_edge_pixels(this->keyEdgeMapValid, this->keyDepthLow, this->keyImgGradx, this->keyImgGrady, this->keyImgGrad, this->keyPixelDataVec);

    // At this location, kd tree construction.
    double invWidth = 1.0 / (double)this->params.calib.width;
    std::vector<std::vector<double>> tmpPixel2Vec;
    tmpPixel2Vec.reserve(0);
    for(int i = 0; i < this->keyPixelDataVec.size(); i++)
    {
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

    // First keyframe is updated done.
    this->isInit = false;
  }
  else // After initial images, successively run the algorithm for the current image.
  {
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
    std::vector<int> rndIdx;
    rndIdx.reserve(0);
    rnd::randsample(this->curPixelDataVec.size(), this->params.hyper.nSample, rndIdx); //sampling without replacement

    // Deprived
    /*std::vector<std::vector<double>> tmpPixel2Vec; // For kdtree generation. Temporary vector container.
    tmpPixel2Vec.reserve(0);
    */

    // Deprived
    /*double invWidth = 1.0 / (double)this->params.calib.width;
    for(int i = 0; i < rndIdx.size(); i++)
    {
      std::vector<double> tmpPixel2;
      tmpPixel2.push_back(this->curPixelDataVec[rndIdx[i]]->u*invWidth);
      tmpPixel2.push_back(this->curPixelDataVec[rndIdx[i]]->v*invWidth);
      tmpPixel2Vec.push_back(tmpPixel2);
    }
    */

    // ====================================================================== //
    // ====================== Iterative optimization ! ====================== //
    // ====================================================================== //
    int    icpIter  = 0;
    double errLast  = 1e9, errPrev = 0;
    double lambda   = 0.05;
    double stepSize = 0.7;

    // initialize the containers which are exploited in the iterative optimization.
    Edgeicp::initialize_pixeldata(this->warpedCurPixelDataVec, this->curPixelDataVec.size()); // warpedCurPixelDataVec ( length : sampled & reduced number ! )
    std::vector<int> refIdx; // reference indices which are matched to the warped current pixels.
    Eigen::MatrixXd J           = Eigen::MatrixXd::Zero(this->params.hyper.nSample,6);
    Eigen::MatrixXd W           = Eigen::MatrixXd::Zero(this->params.hyper.nSample,1);
    Eigen::MatrixXd JW          = Eigen::MatrixXd::Zero(this->params.hyper.nSample,6);
    Eigen::MatrixXd H           = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd HAugmented  = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd diagH       = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd residual    = Eigen::MatrixXd::Zero(this->params.hyper.nSample,1);

    std::cout<<"----------------------------------------------------"<<std::endl;
    while(icpIter < this->params.hyper.maxIter)
    {
      // initialize containers.
      std::vector<std::vector<double>> tmpPixel2Vec; // For kdtree NN search. Temporary vector container.
      tmpPixel2Vec.reserve(0);
      // std::vector<double> residualVec;

      // TODO: warp the current points, ( using "warpedCurPixelDataVec" )
      Edgeicp::warp_pixel_points(this->curPixelDataVec, this->tmpXi, this->warpedCurPixelDataVec);
      // maybe, this->curPixelDataVec to this->warpedCurPixelDataVec is not complete... Due to this, segfault occurs.
      // TODO: reallocate the warped current points to the
      Edgeicp::convert_pixeldatavec_to_vecvec2d(this->warpedCurPixelDataVec, rndIdx, tmpPixel2Vec);

      // TODO: NN search using "warpedCur"
      if(icpIter < this->params.hyper.shiftIter) // 4-D kdtree approximated NN search
      {
        this->keyTree2->kdtree_nearest_neighbor(tmpPixel2Vec, refIdx);
      }
      else // 2-D kdtree search exact NN search
      {
        this->keyTree2->kdtree_nearest_neighbor(tmpPixel2Vec, refIdx);
      }
      // TODO: residual calculation
      Edgeicp::calc_icp_residual_div(this->warpedCurPixelDataVec, this->keyPixelDataVec, rndIdx, refIdx, residual);

      // TODO: t-distribution weight matrix update ( update_t_distribution )
      if(icpIter > 5) // t-distribution weighting after 5 iterations
      {

      }
      /*
       * % Hessian & Jacobian
       * J = calc_ICP_Jacobian_div(ref_pts, warp_pts, g_vec, ref_ind, xi_temp);
       * % residual reweighting
       * W = ones(length(residual),1);
       * if(icp_iter >=5)
       *     W = update_weight_matrix(residual, sig_r,nu);
       * end
       * JW          = bsxfun(@times,J,W);
       * residualW   = bsxfun(@times,residual,W);
       * Hessian     = JW.'*J;
       * delta_xi    = -stepScale*(Hessian + lambda*diag(diag(Hessian)))^-1*JW.'*residual;
       */

      // TODO: calculate Jacobian matrix ( calc_ICP_Jacobian_div )
      Edgeicp::calc_ICP_Jacobian_div(warpedCurPixelDataVec, keyPixelDataVec, rndIdx, refIdx, J);
      // TODO: residual reweighting ( update_weight_matrix )

      // TODO: Weighted residual

      // TODO: Hessian calculation
      H = J.transpose() * J;
      diagH(0,0) = H(0,0);
      diagH(1,1) = H(1,1);
      diagH(2,2) = H(2,2);
      diagH(3,3) = H(3,3);
      diagH(4,4) = H(4,4);
      diagH(5,5) = H(5,5);

      HAugmented = H + lambda*diagH;
      // TODO: delta_xi calculation.
      delXi = -stepSize*H.inverse()*J.transpose()*residual;
      // TODO: xi_temp = xi_temp + delta_xi. update ~
      tmpXi += delXi;
      // TODO: iteration stop condition
      if(0) {
        break;
      }

      icpIter++;
      std::cout<<"--- DEBUG optimization iterations : "<<icpIter<<", wx:"<<tmpXi(0,0)<<", wy:"<<tmpXi(1,0)<<", wz:"<<tmpXi(2,0)<<", vx:"<<tmpXi(3,0)<<", vy:"<<tmpXi(4,0)<<", vz:"<<tmpXi(5,0)<<std::endl;
    }
    std::cout<<std::endl;

    // showing the debuging image.
    if(this->params.debug.imgShowFlag == true)
    {
      cv::Scalar colorLine(0,127,255);
      cv::Scalar colorText(120,120,0);
      cv::Scalar colorCircleRef(0,0,0);
      cv::Scalar colorCircleCur(0,0,255);
      double xr, yr, xc, yc;
      for(int i = 0; i < refIdx.size(); i++)
      {
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

      for(int i = 0; i < this->keyPixelDataVec.size(); i++)
      {
        //xr = this->keyPixelDataVec[i]->u;
        //yr = this->keyPixelDataVec[i]->v;
        //cv::circle(this->keyImgLow, cv::Point(xr, yr), 1, colorCircleRef, CV_FILLED);
      }

      this->curEdgeMap.copyTo(this->debugEdgeImg);
    }



    // If the distance from the current keyframe to the current frame exceeds the threshold, renew the key frame
    if(this->numOfImg % 3 == 1)
    {
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

  // END - algorithm

  // Debug image showing.
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
