#pragma once
#include <vector>    // point datatype
#include <math.h>    // fabs operation
#include "MyHeaps.h" // priority queues
#include "float.h"   // max floating point number
#include "stdint.h"

using namespace std;

typedef std::vector<double> Point;


/// The root node is stored in position 0 of nodesPtrs
#define ROOT 0

/// L2 distance (in dimension ndim) between two points
inline double distance_squared( const std::vector<double>& a, const std::vector<double>& b, double& thres_near){
    double d = 0;
    double d_spatial =0;
    double N = a.size();
   // double lambda = 0.7/4.0;
    double temp;

    if(N==4)
    {
      temp = (a[2]*b[2] + a[3]*b[3]-1.0);
      d_spatial += (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
      //d = d_spatial + 0.175*(a[2]*b[2] + a[3]*b[3]-1)*(a[2]*b[2] + a[3]*b[3]-1);
      return d_spatial + 0.175*temp*temp;
    }
    else
    {
      for( int i=0; i<N; i++ ) d_spatial += (a[i]-b[i])*(a[i]-b[i]);
      return d_spatial;
    }
}

struct Node{
    double key;	///< the key (value along k-th dimension) of the split
    int LIdx;  	///< the index to the left sub-tree (-1 if none)
  	int	RIdx;	  ///< the index to the right sub-tree (-1 if none)
  	int	pIdx;   ///< index of stored data-point (NOTE: only if isLeaf)

    Node(){ LIdx=RIdx=key=pIdx=-1; }
    inline bool isLeaf() const{ return pIdx>=0; }
};

class KDTree{
    /// @{ kdtree constructor/destructor
    public:
        KDTree(){}                           ///< Default constructor (only for load/save)
        KDTree(const std::vector<Point>& points, const double& thres_near_input); ///< tree constructor
        ~KDTree();                           ///< tree destructor
    private:
        int build_recursively(std::vector< std::vector<int> >& sortidx, std::vector<char> &sidehelper, int dim);
        // int heapsort(int dim, std::vector<int>& idx, int len);
    /// @}

    /// @{ basic info
    public:
        inline int size(){ return points.size(); } ///< the number of points in the kd-tree
        inline int ndims(){ return ndim; } ///< the number of dimensions of a point in the kd-tree
    /// @}

    /// @{ core kdtree data
    private:
        int ndim;                 ///< Number of dimensions of the data (>0)
        int npoints;              ///< Number of stored points
        std::vector<Point> points;     ///< Points data, size ?x?
        std::vector<Node*> nodesPtrs;  ///< Tree node pointers, size ?x?
    /// @}

    /// @{ Debuggging helpers
	public:
        void linear_tree_print() const;
        void left_depth_first_print( int nodeIdx=0 ) const;
        void print_tree( int index=0, int level=0 ) const;
        void leaves_of_node( int nodeIdx, std::vector<int>& indexes );
    /// @}


    /// @{ Knn Search & helpers
    public:
        int closest_point(const Point& p);
        void closest_point(const Point &p, int &idx, double &dist);
        void k_closest_points(const Point& Xq, int k, std::vector<int>& idxs, std::vector<double>& distances);
    private:
        void knn_search( const Point& Xq, int nodeIdx = 0, int dim = 0);
        bool ball_within_bounds(const Point& Xq);
        double bounds_overlap_ball(const Point& Xq);
    private:
        int k;					  ///< number of records to search for
        Point Bmin;  		 	  ///< bounding box lower bound
        Point Bmax;  		      ///< bounding box upper bound
        MaxHeap<double> pq;  	  ///< <key,idx> = <distance, node idx>
        bool terminate_search;    ///< true if k points have been found
    /// @}

	/// @{ Points in hypersphere (ball) query
    public:
        void ball_query( const Point& point, const double radius, std::vector<int>& idxsInRange, std::vector<double>& distances );
    private:
        void ball_bbox_query(int nodeIdx, Point& pmin, Point& pmax, std::vector<int>& inrange_idxs, std::vector<double>& distances, const Point& point, const double& radiusSquared, int dim=0);
    /// @}

    /// @{ Range (box) query
    public:
        void range_query( const Point& pmin, const Point& pmax, std::vector<int>& inrange_idxs, int nodeIdx=0, int dim=0 );
    private:
        bool lies_in_range( const Point& p, const Point& pMin, const Point& pMax );
    /// @}

    //Changhyeon function
    public:
      double thres_near;
      void kdtree_nearest_neighbor(const std::vector<std::vector<double>>& query_data, std::vector<int>& ref_ind);

};
