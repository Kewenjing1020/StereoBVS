#include "bilateralSegment.h"

void liftMulFrame(vector<Mat> &imgs,vector<Mat> &disps,vector<Rect> &rects,  Mat &result, int dims)
{
	int nPoints = 0;
	for (int i = 0; i < imgs.size(); ++i)
	{
		nPoints += rects[i].area();
	}
	result = Mat(nPoints, dims, CV_64FC1);
	int startCount = 0;
	for (int i = 0; i<imgs.size(); ++i)
	{
		Mat img_yuv;	
		bgr2yuv(imgs[i], img_yuv);
		int rows = rects[i].height;
		int cols = rects[i].width;
		
		for (int count = 0; count <rows*cols; ++count)
		{
			int y = count%rows;
			int x = count / rows;
			Vec3d *img_data = img_yuv.ptr<Vec3d>(y);
			uchar *disp_data = disps[i].ptr<uchar>(y);
			double *data = result.ptr<double>(count + startCount);
			data[0] = img_data[x][0];
			data[1] = img_data[x][1];
			data[2] = img_data[x][2];
			data[3] = y + rects[i].y;
			data[4] = x + rects[i].x;
			data[5] = disp_data[x];
			data[6] = i;
		}
		startCount += rows*cols;
	}
	double eps = 0.001;
	Mat tempOnes = Mat::ones(result.rows, 1, CV_64FC1);
	for (int c = 0; c < dims; ++c)
	{
		Mat col = result(Range(0, result.rows), Range(c, c + 1));
		double lBound, uBound;
		minMaxIdx(col, &lBound, &uBound);
		lBound -= eps;
		uBound += eps;
		double scaleFactors = (gridSize[c] - 1) *1.0 / (uBound - lBound);
		add(col, -1.0*lBound*tempOnes, col);
		col.convertTo(col, CV_64FC1, scaleFactors, 1.0);
	}	
}

void liftMask(Mat &img, Mat &disp, Mat &result, int dims)
{
	int rows = img.rows;
	int cols = img.cols;
	result = Mat(cols*rows, dims, CV_64FC1);

	Mat img_yuv;		
	bgr2yuv(img, img_yuv);

	for (int count = 0; count <cols*rows; ++count)
	{
		int y = count%rows;
		int x = count / rows;
		Vec3d *img_data = img_yuv.ptr<Vec3d>(y);
		uchar *disp_data = disp.ptr<uchar>(y);
		double *data = result.ptr<double>(count);
		data[0] = img_data[x][0];
		data[1] = img_data[x][1];
		data[2] = img_data[x][2];
		data[3] = y;
		data[4] = x;
		data[5] = disp_data[x];
		data[6] = (1+GRIDSIZE::temporalGridSize)/2.0;
	}
	
	double eps = 0.001;
	Mat tempOnes = Mat::ones(result.rows, 1, CV_64FC1);
	for (int c = 0; c < dims-1; ++c)
	{
		Mat col = result(Range(0, result.rows), Range(c, c + 1));
		double lBound, uBound;
		minMaxIdx(col, &lBound, &uBound);
		lBound -= eps;
		uBound += eps;
		double scaleFactors = (gridSize[c] - 1) *1.0 / (uBound - lBound);
		add(col, -1.0*lBound*tempOnes, col);
		col.convertTo(col, CV_64FC1, scaleFactors, 1.0);
	}
}

// bilateralData must be of type CV_64FC1, bilateralVals must be CV_8UC1
void splat(Mat &bilateralData, Mat &bilateralVals, Mat &splattedData)
{
	int nPoints = bilateralData.rows;
	int nPotentialVertices = prod(gridSize, bilateralData.cols);
	int nClasses = bilateralVals.cols;
	splattedData = Mat::zeros(nPotentialVertices, nClasses, CV_64FC1);
	int dim_exp = 1 << bilateralData.cols;
	//traverse all case of 0/1
	for (uchar i = 1; i <= dim_exp; ++i)
	{
		double tic = getTickCount();
		string bin = dec2bin2(i - 1);

		Mat weights = Mat::ones(nPoints, 1, CV_64FC1);
		int *indices = new int[nPoints];
		
		// calculate the weight & indice of each dim
		for (uchar j = 0; j < bilateralData.cols; ++j)
		{	
			int multiple = prod(gridSize, j);
			if (bin[j]=='0')	//floor
			{				
				//loop on the matrix row
				for (int p = 0; p < nPoints; ++p)
				{
					double temp = bilateralData.at<double>(p, j);
					int fl = floorN(temp);		
					double resi = (1 - temp + fl);
					weights.at<double>(p,0) *= resi;					
					if (j == 0)
						indices[p] = fl - 1;
					else
						indices[p] += multiple*(fl - 1);	//a sum of indice on each dimension
				}
			}
			else //ceil
			{	
				
				for (int p = 0; p < nPoints; ++p)
				{
					double temp = bilateralData.at<double>(p, j);
					int fl = ceilN(temp);
					weights.at<double>(p, 0) *= (temp - floorN(temp));
					if (j == 0)
						indices[p] = fl - 1;
					else
						indices[p] += multiple*(fl - 1);
					
				}
			}
		}
		double toc = getTickCount();
		//accumulate the weight values according to the indices
		//scale to grid size
		for (int c = 0; c < nClasses; ++c)
		{
			accumArray(splattedData, (Mat_<double>)bilateralVals, weights, indices, nPoints, c);	
		}
		delete[] indices;
	}
		
}

int * graphcut(Mat &splattedData, Mat &splattedMask, bool useMask) {

	//filter Null value in splatted data
	vector<int> occupiedVertices;	//fewer occupied Vertices than in matlab, since the values are 0.002 smaller
	int reduceSize=0;

	findNonNull(occupiedVertices, splattedData);	//store the index of the non-zero vertex
	reduceSize = occupiedVertices.size();
	Mat splattedData_sml = Mat::zeros(reduceSize, 1, CV_64FC1);
	Mat splattedMask_sml = Mat::zeros(reduceSize, 2, CV_64FC1);	
	transValue(splattedData, splattedData_sml, occupiedVertices);
	transValue(splattedMask, splattedMask_sml, occupiedVertices);

	//Build pairwise cost matrix
	int size[] = { reduceSize, reduceSize };
	SparseMat B = SparseMat(2, size, CV_64FC1);
	createAdjacencyMatrix(B, occupiedVertices, splattedData_sml);
	// solve GraphCut
	Mat dataCost;
	if(!useMask)
		splattedMask_sml.convertTo(dataCost, CV_32FC1, unaryWeight_disp);
	else
		splattedMask_sml.convertTo(dataCost, CV_32FC1, unaryWeight_initMask);
	Mat smoothCost = (Mat)Matx22f(0, pairwiseWeight,
		pairwiseWeight, 0);
	int *L = graph(dataCost, smoothCost, B);
	int *labels = new int[prod(gridSize, DIM_OF_FEATURE)];
	for (int i = 0; i<occupiedVertices.size(); ++i)
	{
		labels[occupiedVertices[i]] = L[i];
	}
	delete[] L;
	return labels;
}

void createAdjacencyMatrix(SparseMat &B, vector <int> &occupiedVertices, Mat &splattedData) {
	double minGraphWeight = 0.01;

	Mat occupiedVertx_mat;
	vect2mat(occupiedVertx_mat, occupiedVertices);
	SparseMat A;
	for (int i = 0; i < DIM_OF_FEATURE; ++i) {
		if (dimensionWeights[i] < EPS) {
			continue;
		}
		else {
			// find the index offset
			int offset = (i == 0) ? 1 : prod(gridSize, i);

			//compute neighbor indices in the graph vertex space
			Mat leftIndices, rightIndices;
			Mat tempOnes = Mat::ones(occupiedVertx_mat.rows, 1, CV_32S);
			scaleAdd(tempOnes, -1*offset, occupiedVertx_mat, leftIndices);
			scaleAdd(tempOnes, offset, occupiedVertx_mat, rightIndices);

			//project onto the front dimension
			int maxidx = prod(gridSize, i+1);
			Mat centerModulo = Mat::zeros(occupiedVertx_mat.rows, 1, CV_32S);
			for (int j = 0; j < occupiedVertx_mat.rows; ++j) {
				int tempdata = occupiedVertx_mat.at<int>(j,0);
				tempdata = (tempdata - 1) / maxidx * maxidx;
				centerModulo.at<int>(j, 0) = tempdata;
			}
			
			//check if out of bounds
			for (int j = 0; j < leftIndices.rows; ++j) {
				if ((leftIndices.at<int>(j, 0) - centerModulo.at<int>(j, 0)) < 1) {
					leftIndices.at<int>(j, 0) = 0;
				}
				if ((rightIndices.at<int>(j, 0) - centerModulo.at<int>(j, 0)) > maxidx ){
					rightIndices.at<int>(j, 0) = 0;
				}
			}
			
			// Convert the indices into the occupied Vertex space
			// the indices are sorted from lowest to highest
			Mat leftIndices2, leftCenterIndices;
			intersect(leftIndices, occupiedVertx_mat, offset, leftIndices2, leftCenterIndices);

			Mat rightIndices2, rightCenterIndices;
			intersect(occupiedVertx_mat, rightIndices, offset, rightCenterIndices, rightIndices2);

			// weight for an edge is the sum of the vertex weights
			Mat wLeft = Mat::zeros(leftIndices2.rows, 1, CV_64FC1);
			sumWeight(splattedData, leftCenterIndices, wLeft);
			sumWeight(splattedData, leftIndices2, wLeft);

			Mat wRight = Mat::zeros(leftIndices2.rows, 1, CV_64FC1);;
			sumWeight(splattedData, rightCenterIndices, wRight);
			sumWeight(splattedData, rightIndices2, wRight);

			Mat sp_v1 = Mat::ones(leftCenterIndices.rows, 1, CV_64FC1);
			cv::multiply(sp_v1, wLeft, sp_v1, dimensionWeights[i]);
			max(sp_v1, minGraphWeight, sp_v1);
			Mat sp_v2 = Mat::ones(rightCenterIndices.rows, 1, CV_64FC1);
			cv::multiply(sp_v2, wRight, sp_v2, dimensionWeights[i]);
			max(sp_v2, minGraphWeight, sp_v2);
			for (int j = 0; j < leftCenterIndices.rows; ++j) {
				int x = leftCenterIndices.at<int>(j, 0);
				int y = leftIndices2.at<int>(j, 0);

				B.ref<double>(x, y) += sp_v1.at<double>(j, 0);
				int x2 = rightCenterIndices.at<int>(j, 0);
				int y2 = rightIndices2.at<int>(j, 0);				
				B.ref<double>(x2, y2) += sp_v2.at<double>(j, 0);

			}
		}
	}
}

int * graph(Mat &dataCost, Mat &smoothCost, SparseMat &adj)
{
	int num_pixels = adj.size()[0];
	int num_labels = dataCost.cols;
	int *result = new int[num_pixels];   // stores result of optimization

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels, num_labels);
		float *dc = dataCost.ptr<float>(0);
		float *sc = smoothCost.ptr<float>(0);
		gc->setDataCost(dc);
		gc->setSmoothCost(sc);
		SparseMatConstIterator_<double>	it = adj.begin<double>(), it_end = adj.end<double>();

		for (; it != it_end; ++it)
		{
			const SparseMat::Node* n = it.node();
			int y = n->idx[0];
			int x = n->idx[1];
			float cost = (*it) *adjWeight;
			if( y < x)
				gc->setNeighbors(y, x, cost);
		}
		gc->expansion(500);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);

		for (int i = 0; i < num_pixels; i++)
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e) {
		e.Report();
	}
	return result;
}

void sliceLabels(Mat &labels, Mat &bilateralData, Mat &sliced) {
	int nPoints = bilateralData.rows;
	//cout << " sliced nPoints: " << nPoints << endl;
	int nDims = bilateralData.cols;
	int nVertices = labels.rows;
	int nClasses = labels.cols;
	sliced = Mat::zeros(nPoints, nClasses, CV_64FC1);
	int dim_exp = 1 << bilateralData.cols;//2^n
	for (uchar i = 1; i <= dim_exp; ++i)
	{
		//cout << "at each iteration... " << endl;
		double tic = getTickCount();
		//bool *bin = dec2bin(i - 1);
		string bin = dec2bin2(i - 1);

		Mat weights = Mat::ones(nPoints, 1, CV_64FC1);
		int *indices = new int[nPoints];

		// calculate the weight & indice of each dim
		for (uchar j = 0; j < bilateralData.cols; ++j)
		{
			int multiple = prod(gridSize, j);
			//vector<double> remainders;
			if (bin[j] == '0')	//floor
			{
				//loop on the matrix row
				double tic = getTickCount();
				for (int p = 0; p < nPoints; ++p)
				{
					double temp = bilateralData.at<double>(p, j);
					int fl = floorN(temp);
					//remainders.push_back(temp - fl);
					double resi = (1 - temp + fl);
					weights.at<double>(p, 0) *= resi;
					if (j == 0)
						indices[p] = fl - 1;
					else
						indices[p] += multiple*(fl - 1);	//a sum of indice on each dimension
				}
				/*cout << "Iteration of weights computation for one dimension takes "<<(getTickCount() - tic) / getTickFrequency() << " s" << endl;*/
			}
			else //ceil
			{

				for (int p = 0; p < nPoints; ++p)
				{
					double temp = bilateralData.at<double>(p, j);
					int fl = ceilN(temp);
					weights.at<double>(p, 0) *= (temp - floorN(temp));
					if (j == 0)
						indices[p] = fl - 1;
					else
						indices[p] += multiple*(fl - 1);

				}
			}
		}

		for (int index = 0; index < nPoints; ++index)
		{			
			double temp = weights.at<double>(index, 0)*labels.at<int>(indices[index], 0);
			sliced.at<double>(index, 0) += temp;	
		}
		delete[] indices;
	}
}
