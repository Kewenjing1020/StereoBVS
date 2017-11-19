#pragma once
#include "common.h"
#include "GCoptimization.h"
#include <fstream>

#define DIM_OF_FEATURE 7
#define EPS 1e-14

#ifdef LOW_RESO
const struct GRIDSIZE
{
	const static int intensityGridSize = 3;// 5;// 7;
	const static int chromaGridSize = 5;// 10;
	const static int spatialGridSize = 20;// 15;
	const static int temporalGridSize = 2;
	const static int dispGridSize = 2;
};
#endif // LOW_RESO

#ifndef LOW_RESO
const struct GRIDSIZE
{
	const static int intensityGridSize = 7;
	const static int chromaGridSize = 9;
	const static int spatialGridSize = 13;
	const static int temporalGridSize = 2;
	const static int dispGridSize = 2;
};
#endif // LOW_RESO

//length of each block of frames
const static int maxFrame = 2;
const static float roiRatio = 1/3.f;
//Graph Cut Parameters
const static float adjWeight = 1;
const static float	pairwiseWeight = 1;
const static float unaryWeight_disp = 2;
const static float unaryWeight_initMask = 1e3; 


const static float dispWeight = 1;
const static double temporalWeight = 1e5;
const static double intensityWeight = 0.05;
const static double colorWeight = 0.03;
const static double spatialWeight = 0.3;
const static double minGraphWeight = 0.001;

const int gridSize[DIM_OF_FEATURE] = { GRIDSIZE::intensityGridSize,GRIDSIZE::chromaGridSize, GRIDSIZE::chromaGridSize, GRIDSIZE::spatialGridSize,GRIDSIZE::spatialGridSize,GRIDSIZE::dispGridSize, GRIDSIZE::temporalGridSize };
const double dimensionWeights[DIM_OF_FEATURE] = { colorWeight, colorWeight, colorWeight, spatialWeight, spatialWeight, dispWeight, temporalWeight };


static void bgr2yuv(Mat &src, Mat &dst)
{
	Mat src_n, src_g;
	blur(src, src_g, Size(3, 3));
	src_g.convertTo(src_n, CV_64FC3, 1.0 / 255);	
	dst = Mat(src.rows, src.cols, CV_64FC3);
	for (int i = 0; i < src.rows; ++i)
	{
		Vec3d *bgr = src_n.ptr<Vec3d>(i);
		Vec3d *yuv = dst.ptr<Vec3d>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			yuv[j][0] = 0.299*bgr[j][2] + 0.587*bgr[j][1] + 0.114*bgr[j][0];
			yuv[j][1] = 0.596*bgr[j][2] - 0.274*bgr[j][1] - 0.322*bgr[j][0];
			yuv[j][2] = 0.211*bgr[j][2] - 0.523*bgr[j][1] + 0.312*bgr[j][0];
		}
	}
}

static int prod(const int (&v)[DIM_OF_FEATURE], int k)
{
	
	if (k > DIM_OF_FEATURE || k < 0) {
		exception ("invalid parameter");
		return 0;
	}
		
	int result = 1;
	for (int i = 0; i < k; ++i)	
	{
		result *= v[i];
	}
	
	return result;
}


static string dec2bin2(const int n) {
	int i = n;
	string bin("0", DIM_OF_FEATURE);
	int reminder;
	for (int j = 0; j < DIM_OF_FEATURE; ++j) {
		bin[DIM_OF_FEATURE - 1 - j] = '0';
		if (i % 2 != 0)
				bin[DIM_OF_FEATURE -1 - j] = '1';
		i /= 2;
	}
	return bin;
}

static int floorN(double n)
{
	return floor(n);
}

static int ceilN(double n)
{
	return ceil(n);
}

static void accumArray(Mat &splatData, Mat &bilateralValue,  Mat &weights, int *indice, int nPoints, int c) {

	if (c > splatData.cols || c > bilateralValue.cols) {
		exception("c invalid!");
	}

	//create a new matrice of lenth (indiceMax), to store weights
	Mat accumData= Mat::zeros(splatData.rows, 1,  CV_64FC1);

	//accumulate weights to its relevant indice
	for (int j = 0; j < nPoints; j++) {
		int curIndice = indice[j];
		accumData.at<double>(curIndice, 0) += weights.at<double>(j, 0)* bilateralValue.at<uchar>(j, c);
	}

	//add to splatedData
	for (int k = 0; k < splatData.rows; ++k) {
		//splattedVal[k] += data2[k];
		splatData.at<double>(k, c) += accumData.at<double>(k, 0);
	}

}

static void accumArray(Mat &splatData, Mat_<double> &bilateralValue, Mat &weights, int *indice, int nPoints, int c) {

	if (c > splatData.cols || c > bilateralValue.cols) {
		exception("c invalid!");
	}

	//create a new matrice of lenth (indiceMax), to store weights
	Mat accumData = Mat::zeros(splatData.rows, 1, CV_64FC1);

	//accumulate weights to its relevant indice
	for (int j = 0; j < nPoints; j++) {
		int curIndice = indice[j];
		accumData.at<double>(curIndice, 0) += weights.at<double>(j, 0)* bilateralValue(j,c);
	}

	//add to splatedData
	for (int k = 0; k < splatData.rows; ++k) {
		//splattedVal[k] += data2[k];
		splatData.at<double>(k, c) += accumData.at<double>(k, 0);
	}

}


static void binMask(Mat &mask, Mat &mask_bin) {
	// maskValues = cat(2,mask(:)~=0.,mask(:)==0);
	int area2 = mask.rows *mask.cols;	
	mask_bin = Mat(area2, 2, CV_8UC1, Scalar(0));
	Mat img_t = mask.t();
	for (int j = 0; j < img_t.rows; ++j) {
		const uchar *data = img_t.ptr<uchar>(j);
		for (int k = 0; k < img_t.cols; ++k) {
			uchar temp = data[k];
			int index = j * img_t.cols + k;
			if (temp >= 200) {
				mask_bin.at<uchar>(index, 0) = 1;
				//cout << index << endl;
			}
			else {
				mask_bin.at<uchar>(index, 1) = 1;
			}
		}
	}
	
}

static void findNonNull(vector<int> &occupiedVertices, Mat &splattedData) {
	
	for (int i = 0; i < splattedData.rows; ++i) {	
		const double* data = splattedData.ptr<double>(i);
		for (int j = 0; j < splattedData.cols; ++j) {
			double x = data[j];	
			if (x>EPS) {
				occupiedVertices.push_back(i); // splattedData.cols=1
			}
		}
		
	}
	int points = occupiedVertices.size();
	if (points == 0) {
		exception("find non occupied vertice!");
	}

}

static void transValue(Mat &splattedData, Mat &splattedData_use, vector<int> &occupiedVertices) {
	// remain the occupied splattedData
	for (int i = 0; i < occupiedVertices.size(); ++i) {
		int index = occupiedVertices[i];
		for (int j = 0; j < splattedData.cols; ++j) {
			splattedData_use.at<double>(i, j) = splattedData.at<double>(index, j);
		}

	}
}


static void vect2mat(Mat &a, vector<int> &b) {
	a = Mat::zeros(b.size(), 1, CV_32S);
	for (int i = 0; i < b.size(); ++i) {
		a.at<int>(i, 0) = b[i];
	}
}

static void intersect(Mat &A, Mat &B, int offset, Mat &ai, Mat &bi) {
	// find C=common(A,B), C=A(ai)=B(bi)
	// A<B
	vector<int> a;
	vector<int> b;
	int j = 0;
	for (int i = 1; i < A.rows; ++i) {
		if (j == B.rows)
			break;
		if (A.at<int>(i, 0) == B.at<int>(j, 0)) {
			a.push_back(i);
			b.push_back(j);
			++j;
		}
		else if(A.at<int>(i, 0) < B.at<int>(j, 0)){
			continue;
		}
		else {
			while (A.at<int>(i, 0) > B.at<int>(j, 0)&& j<B.rows-1) {
				++j;
				if (A.at<int>(i, 0) == B.at<int>(j, 0)) {
					a.push_back(i);
					b.push_back(j);
					break;
				}
			}
		}
		
	}
	vect2mat(ai, a);
	vect2mat(bi, b);
}

static void sumWeight(Mat &vertexWeights, Mat &indices, Mat &sum) {
	for (int i = 0; i < indices.rows; ++i) {
		int j = indices.at<int>(i, 0);
		sum.at<double>(i, 0) += vertexWeights.at<double>(j, 0);
	}
}

//void liftFrame(Mat &img, Mat &disp, Rect &rect, Mat &result, int dims = DIM_OF_FEATURE - 1);
void liftMask(Mat &img, Mat &disp, Mat &result, int dims = DIM_OF_FEATURE);
void liftMulFrame(vector<Mat> &imgs, vector<Mat> &disps, vector<Rect> &rects, Mat &result, int dims = DIM_OF_FEATURE);
void splat(Mat &bilateralData, Mat &bilateralVals, Mat &splattedData);
int * graphcut(Mat &splattedData, Mat &splattedMask, bool useMask = false);
void createAdjacencyMatrix(SparseMat &B, vector <int> &occupiedVertices, Mat &splattedData);
int * graph(Mat &dataCost, Mat &smoothCost, SparseMat &adj);
void sliceLabels(Mat &labels, Mat &bilateralData, Mat &sliced);

static void loadMatFromTxt(const char *filename, Mat &m, int rows, int cols, int type)
{
	m = Mat(rows, cols, type);
	ifstream f(filename);
	if (type == CV_32FC1)
	{
		for (int y = 0; y < rows; ++y)
		{
			float *data = m.ptr<float>(y);
			for (int x = 0; x < cols; ++x)
			{
				float temp;
				f >> temp;
				data[x] = temp;
			}
		}
	}
	else if (type == CV_32SC1)
	{
		for (int y = 0; y < rows; ++y)
		{
			int *data = m.ptr<int>(y);
			for (int x = 0; x < cols; ++x)
			{
				int temp;
				f >> temp;
				data[x] = temp;
			}
		}
	}
	else if (type == CV_64FC1)
	{
		for (int y = 0; y < rows; ++y)
		{
			double *data = m.ptr<double>(y);
			for (int x = 0; x < cols; ++x)
			{
				double temp;
				f >> temp;
				data[x] = temp;
			}
		}
	}
	else if (type == CV_8UC1)
	{
		for (int y = 0; y < rows; ++y)
		{
			uchar *data = m.ptr<uchar>(y);
			for (int x = 0; x < cols; ++x)
			{
				uchar temp;
				f >> temp;
				data[x] = temp;
			}
		}
	}

}
