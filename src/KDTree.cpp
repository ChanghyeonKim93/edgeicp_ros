#include "KDTree.h"
//----------------------------------------------------------------------------------------
//
//                                  Implementation
//
//----------------------------------------------------------------------------------------

/**
 * Creates a KDtree filled with the provided data.
 *
 * @param points   a std::vector< std::vector<double> > containing the point data
 * 				   the number of points and the dimensionality is inferred
 *                 by the data
 */
KDTree::KDTree(const std::vector<Point>& points,const double& thres_near_input){
    // initialize data
    this -> npoints   = points.size();
    this -> ndim      = points[0].size();
    this -> points    = points;
    this -> thres_near= thres_near_input;
    nodesPtrs.reserve( npoints );

    // used for sort-based tree construction
    // tells whether a point should go to the left or right
    // array in the partitioning of the sorting array
    std::vector<char> sidehelper(npoints,'x');

    // Invoke heap sort generating indexing std::vectors
    // sorter[dim][i]: in dimension dim, which is the i-th smallest point?
    std::vector< MinHeap<double> > heaps(ndim, npoints);
    for( int dIdx=0; dIdx<ndim; dIdx++ )
        for( int pIdx=0; pIdx<npoints; pIdx++ )
            heaps[dIdx].push( points[pIdx][dIdx], pIdx );
    std::vector< std::vector<int> > sorter( ndim, std::vector<int>(npoints,0) );
    for( int dIdx=0; dIdx<ndim; dIdx++ )
        heaps[dIdx].heapsort( sorter[dIdx] );

    build_recursively(sorter, sidehelper, 0);
}

KDTree::~KDTree(){
    for (unsigned int i=0; i < nodesPtrs.size(); i++)
        delete nodesPtrs[i];
}


/**
 * Algorithm that recursively performs median splits along dimension "dim"
 * using the pre-prepared information given by the sorting.
 *
 * @param sortidx: the back indexes produced by sorting along every dimension used for median computation
 * @param pidx:    a std::vector of indexes to active elements
 * @param dim:     the current split dimension
 *
 * @note this is the memory-friendly version
 */
void print_sorter(const char* message, std::vector< std::vector<int> >& srtidx){
    cout << message << endl;
    for (unsigned int j=0; j <srtidx.size(); j++){
        for (unsigned int i=0; i < srtidx[j].size(); i++)
            cout << srtidx[j][i] << " ";
        cout << endl;
    }
}
int KDTree::build_recursively(std::vector< std::vector<int> >& sorter, std::vector<char>& sidehelper, int dim){
    // Current number of elements
    int numel = sorter[dim].size();

    // Stop condition
    if(numel == 1) {
        Node *node = new Node();		// create a new node
        int nodeIdx = nodesPtrs.size(); // its address is
        nodesPtrs.push_back( node ); 	// important to push back here
        node->LIdx = -1;				// no child
        node->RIdx = -1;    			// no child
        /// @todo take it from sorter
        node->pIdx = sorter[dim][0];    // the only index available
        node->key = 0;					// key is useless here
        return nodeIdx;
    }

    // defines median offset
    // NOTE: pivot goes to the LEFT sub-array
    int iMedian = floor((numel-1)/2.0);
    int pidxMedian = sorter[dim][iMedian];
    int nL = iMedian+1;
    int nR = numel-nL;

    // Assign l/r sides
    for(int i=0; i<sorter[dim].size(); i++){
        int pidx = sorter[dim][i];
        sidehelper[ pidx ] = (i<=iMedian) ? 'l':'r';
    }

    // allocate the std::vectors initially with invalid data
    std::vector< std::vector<int> > Lsorter(ndim, std::vector<int>(nL,-1));
    std::vector< std::vector<int> > Rsorter(ndim, std::vector<int>(nR,-1));

    for(int idim=0; idim<ndims(); idim++){
        int iL=0, iR=0;
        for(int i=0; i<sorter[idim].size(); i++){
            int pidx = sorter[idim][i];
            if(sidehelper[pidx]=='l')
                Lsorter[idim][iL++] = pidx;
            if(sidehelper[pidx]=='r')
                Rsorter[idim][iR++] = pidx;
        }
    }

#if DEBUG
    if(numel>2){
        cout << "---- SPLITTING along " << dim << endl;
        print_sorter("original: ", sorter);
        print_sorter("L: ", Lsorter);
        print_sorter("R: ", Rsorter);
    }
#endif

    // CREATE THE NODE
    Node* node = new Node();
    int nodeIdx = nodesPtrs.size(); //size() is the index of last element+1!!
    nodesPtrs.push_back( node ); //important to push back here
    node->pIdx  	= -1; //not a leaf
    node->key  		= points[ pidxMedian ][ dim ];
    node->LIdx 		= build_recursively( Lsorter, sidehelper, (dim+1)%ndim );
    node->RIdx 		= build_recursively( Rsorter, sidehelper, (dim+1)%ndim );
    return nodeIdx;
}

/**
 * Prints the tree traversing linearly the structure of nodes
 * in which the tree is stored.
 */
void KDTree::linear_tree_print() const{
    for (unsigned int i=0; i < nodesPtrs.size(); i++) {
        Node* n = nodesPtrs[i];

    }
}

/**
 * Prints the tree in depth first order, visiting
 * the node to the left, then the root, then the node
 * to the right recursively.
 *
 * @param nodeIdx the node of the index from which to start printing
 *        (default is the root)
 */
void KDTree::left_depth_first_print( int nodeIdx /*=0*/) const{
    Node* currnode = nodesPtrs[nodeIdx];

    if( currnode -> LIdx != -1 )
        left_depth_first_print( currnode -> LIdx );
    cout << currnode -> key << " ";
    if( currnode -> RIdx != -1 )
        left_depth_first_print( currnode -> RIdx );
}

/**
 * Prints the tree in a structured way trying to make clear
 * the underlying hierarchical structure using indentation.
 *
 * @param index the index of the node from which to start printing
 * @param level the key-dimension of the node from which to start printing
 */
void KDTree::print_tree( int index/*=0*/, int level/*=0*/ ) const{
    Node* currnode = nodesPtrs[index];

    // leaf
    if( currnode->pIdx >= 0 ){
        cout << "--- "<< currnode->pIdx+1 << " --- "; //node is given in matlab indexes
        for( int i=0; i<ndim; i++ ) cout << points[ currnode->pIdx ][ i ] << " ";
        cout << endl;
    }
    else
        cout << "l(" << level%ndim << ") - " << currnode->key << " nIdx: " << index << endl;

    // navigate the childs
    if( currnode -> LIdx != -1 ){
        for( int i=0; i<level; i++ ) cout << "  ";
        cout << "left: ";
        print_tree( currnode->LIdx, level+1 );
    }
    if( currnode -> RIdx != -1 ){
        for( int i=0; i<level; i++ ) cout << "  ";
        cout << "right: ";
        print_tree( currnode->RIdx, level+1 );
    }
}

/**
 * k-NN query: computes the k closest points in the database to a given point
 * and returns their indexes.
 *
 * @param Xq            the query point
 * @param k             the number of neighbors to search for
 * @param idxs          the search results
 * @param distances     the distances from the points
 *
 */
void KDTree::k_closest_points(const Point& Xq, int k, std::vector<int>& idxs, std::vector<double>& distances){
    // initialize search data
    Bmin = std::vector<double>(ndim,-DBL_MAX);
    Bmax = std::vector<double>(ndim,+DBL_MAX);
    this->k = k;
    this->terminate_search = false;

    // call search on the root [0] fill the queue
    // with elements from the search
    knn_search( Xq );

    // scan the created pq and extract the first "k" elements
    // pop the remaining
    int N = pq.size();
    for (int i=0; i < N; i++) {
        pair<double, int> topel = pq.top();
        pq.pop();
        if( i>=N-k ){
            idxs.push_back( topel.second );
            distances.push_back( sqrt(topel.first) ); // it was distance squared
        }
    }

    // invert the std::vector, passing first closest results
    std::reverse( idxs.begin(), idxs.end() );
    std::reverse( distances.begin(), distances.end() );
}

/**
 * The algorithm that computes kNN on a k-d tree as specified by the
 * referenced paper.
 *
 * @param nodeIdx the node from which to start searching (default root)
 * @param Xq the query point
 * @param dim the dimension of the current node (default 0, the first)
 *
 * @note: this function and its subfunctions make use of shared
 *        data declared within the data structure: Bmin, Bmax, pq
 *
 * @article{friedman1977knn,
 *          author = {Jerome H. Freidman and Jon Louis Bentley and Raphael Ari Finkel},
 *          title = {An Algorithm for Finding Best Matches in Logarithmic Expected Time},
 *          journal = {ACM Trans. Math. Softw.},
 *          volume = {3},
 *          number = {3},
 *          year = {1977},
 *          issn = {0098-3500},
 *          pages = {209--226},
 *          doi = {http://doi.acm.org/10.1145/355744.355745},
 *          publisher = {ACM},
 *          address = {New York, NY, USA}}
 */
void KDTree::knn_search( const Point& Xq, int nodeIdx/*=0*/, int dim/*=0*/){
    // cout << "at node: " << nodeIdx << endl;
    Node* node = nodesPtrs[ nodeIdx ];
    double temp;

    // We are in LEAF
    if( node -> isLeaf() ){
        double distance = distance_squared( Xq, points[ node->pIdx ],this->thres_near);

        // pqsize is at maximum size k, if overflow and current record is closer
        // pop further and insert the new one
        if( pq.size()==k && pq.top().first>distance ){
            pq.pop(); // remove farther record
            pq.push( distance, node->pIdx ); //push new one
        }
        else if( pq.size()<k )
            pq.push( distance, node->pIdx );

        return;
    }

    ////// Explore the sons //////
    // recurse on closer son
    if( Xq[dim] <= node->key ){
        temp = Bmax[dim]; Bmax[dim] = node->key;
        knn_search( Xq, node->LIdx, (dim+1)%ndim );
        Bmax[dim] = temp;
    }
    else{
        temp = Bmin[dim]; Bmin[dim] = node->key;
        knn_search( Xq, node->RIdx, (dim+1)%ndim );
        Bmin[dim] = temp;
    }
    // recurse on farther son
    if( Xq[dim] <= node->key ){
        temp = Bmin[dim]; Bmin[dim] = node->key;
        if( bounds_overlap_ball(Xq) )
            knn_search( Xq, node->RIdx, (dim+1)%ndim );
        Bmin[dim] = temp;
    }
    else{
        temp = Bmax[dim]; Bmax[dim] = node->key;
        if( bounds_overlap_ball(Xq) )
            knn_search( Xq, node->LIdx, (dim+1)%ndim );
        Bmax[dim] = temp;
    }
}

void KDTree::leaves_of_node( int nodeIdx, std::vector<int>& indexes ){
    Node* node = nodesPtrs[ nodeIdx ];
    if( node->isLeaf() ){
        indexes.push_back( node->pIdx );
        return;
    }

    leaves_of_node( node->LIdx, indexes );
    leaves_of_node( node->RIdx, indexes );
}

void KDTree::closest_point(const Point &p, int& idx, double& dist){
    std::vector<int> idxs;
    std::vector<double> dsts;
    k_closest_points(p,1,idxs,dsts);
    idx = idxs[0];
    dist = dsts[0];
    return;
}

int KDTree::closest_point(const Point &p){
    int idx;
    double dist;
    closest_point(p,idx,dist);
    return idx;
}

void KDTree::kdtree_nearest_neighbor(const std::vector<std::vector<double>>& query_data, std::vector<int>& ref_ind){ // changhyeon kim update
  int npoints = query_data.size();
  int ndims = query_data[0].size();
  std::vector<double> query (ndims,0);
  //std::cout<<"npoints : " <<npoints <<", ndims : "<<ndims<<std::endl; // for debug
  int idx;
  double dist;

  std::vector<int> null_vec;
  ref_ind.swap(null_vec);

  for(int i=0;i<npoints;i++){
    for(int j=0; j<ndims; j++) query[j] = query_data[i][j];
    closest_point(query,idx,dist);//find closest point.
  //  std::cout<<idx<<std::endl;
    if(dist==98765.0)
    {
      ref_ind.push_back(-1);
      std::cout<<"index over"<<std::endl;
    }
    else ref_ind.push_back(idx);
    // ref_ind.push_back(idx);
  }
  //std::cout<<"ref : "<<ref_ind.size()<<std::endl;
}

/** @see knn_search
 * this function was in the original paper implementation.
 * Was this function useful? How to implement the "done"
 * as opposed to "return" was a mistery. It was used to
 * interrupt search. It might be worth to check its purpose.
 *
 * Verifies if the ball centered in the query point, which
 * radius is the distace from the sample Xq to the k-th best
 * found point, doesn't touches the boundaries of the current
 * BBox.
 *
 * @param Xq the query point
 * @return true if the search can be safely terminated, false otherwise
 */
bool KDTree::ball_within_bounds(const Point& Xq){

    //extract best distance from queue top
    double best_dist = sqrt( pq.top().first );
    // check if ball is completely within BBOX
    for (int d=0; d < ndim; d++)
        if( fabs(Xq[d]-Bmin[d]) < best_dist || fabs(Xq[d]-Bmax[d]) < best_dist )
            return false;
    return true;
}
/** @see knn_search
 *
 * This is the search bounding condition. It checks wheter the ball centered
 * in the sample point, with radius given by the k-th closest point to the query
 * (if k-th closest not defined is \inf), touches the bounding box defined for
 * the current node (Bmin Bmax globals).
 *
 */
double KDTree::bounds_overlap_ball(const Point& Xq){
    // k-closest still not found. termination test unavailable
    if( pq.size()<k )
        return true;

    double sum = 0;
    //extract best distance from queue top
    double best_dist_sq = pq.top().first;
    // cout << "current best dist: " << best_dist_sq << endl;
    for (int d=0; d < ndim; d++) {
        // lower than low boundary
        if( Xq[d] < Bmin[d] ){
            sum += ( Xq[d]-Bmin[d] )*( Xq[d]-Bmin[d] );
            if( sum > best_dist_sq )
                return false;
        }
        else if( Xq[d] > Bmax[d] ){
            sum += ( Xq[d]-Bmax[d] )*( Xq[d]-Bmax[d] );
            if( sum > best_dist_sq )
                return false;
        }
        // else it's in range, thus distance 0
    }

    return true;
}


/**
 * Query all points at distance less or than radius from point
 *
 * @param point the center of the ndim dimensional query ball
 * @param radius the radius of the ndim dimensional query ball
 * @param idxsInRange (return) a collection of indexes of points that fall within
 *        the given ball.
 * @param distances the distances from the query point to the points within the ball
 *
 * @note This is a fairly unefficient implementation for two reasons:
 *       1) the range query is not implemented in its most efficient way
 *       2) all the points in between the bbox and the ball are visited as well, then rejected
 */
void KDTree::ball_query( const Point& point, const double radius, std::vector<int>& idxsInRange, std::vector<double>& distances ){
    // create pmin pmax that bound the sphere
    Point pmin(ndim,0);
    Point pmax(ndim,0);
    for (int dim=0; dim < ndim; dim++) {
        pmin[dim] = point[dim]-radius;
        pmax[dim] = point[dim]+radius;
    }
    // start from root at zero-th dimension
    ball_bbox_query( ROOT, pmin, pmax, idxsInRange, distances, point, radius*radius, 0 );
}
/** @see ball_query, range_query
 *
 * Returns all the points withing the ball bounding box and their distances
 *
 * @note this is similar to "range_query" i just replaced "lies_in_range" with "euclidean_distance"
 */
void KDTree::ball_bbox_query(int nodeIdx, Point& pmin, Point& pmax, std::vector<int>& inrange_idxs, std::vector<double>& distances, const Point& point, const double& radiusSquared, int dim/*=0*/){
    Node* node = nodesPtrs[nodeIdx];

    // if it's a leaf and it lies in R
    if( node->isLeaf() ){
        double distance = distance_squared(points[node->pIdx], point,this->thres_near);
        if( distance <= radiusSquared ){
            inrange_idxs.push_back( node->pIdx );
            distances.push_back( sqrt(distance) );
            return;
        }
    }
    else{
        if(node->key >= pmin[dim] && node->LIdx != -1 )
            ball_bbox_query( node->LIdx, pmin, pmax, inrange_idxs, distances, point, radiusSquared, (dim+1)%ndim);
        if(node->key <= pmax[dim] && node->RIdx != -1 )
            ball_bbox_query( node->RIdx, pmin, pmax, inrange_idxs, distances, point, radiusSquared, (dim+1)%ndim);
    }
}

/**
 * k-dimensional Range query: given a bounding box in ndim dimensions specified by the parameters
 * returns all the indexes of points within the bounding box.
 *
 * @param pmin the lower corner of the bounding box
 * @param pmax the upper corner of the bounding box
 * @param inrange_idxs the indexes which satisfied the query, falling in the bounding box area
 *
 */
void KDTree::range_query( const Point& pmin, const Point& pmax, std::vector<int>& inrange_idxs, int nodeIdx/*=0*/, int dim/*=0*/ ){
    Node* node = nodesPtrs[nodeIdx];
    //cout << "I am in: "<< nodeIdx << "which is is leaf?" << node->isLeaf() << endl;

    // if it's a leaf and it lies in R
    if( node->isLeaf() ){
        if( lies_in_range(points[node->pIdx], pmin, pmax) ){
            inrange_idxs.push_back( node->pIdx );
            return;
        }
    }
    else{
        if(node->key >= pmin[dim] && node->LIdx != -1 )
            range_query( pmin, pmax, inrange_idxs, node->LIdx, (dim+1)%ndim);
        if(node->key <= pmax[dim] && node->RIdx != -1 )
            range_query( pmin, pmax, inrange_idxs, node->RIdx, (dim+1)%ndim);
    }
}
/** @see range_query
 * Checks if a point lies in the bounding box (defined by pMin and pMax)
 *
 * @param p the point to be checked for
 * @param pMin the lower corner of the bounding box
 * @param pMax the upper corner of the bounding box
 *
 * @return true if the point lies in the box, false otherwise
 */
bool KDTree::lies_in_range( const Point& p, const Point& pMin, const Point& pMax ){
    for (int dim=0; dim < ndim; dim++)
        if( p[dim]<pMin[dim] || p[dim]>pMax[dim] )
            return false;
    return true;
}
