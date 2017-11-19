#include "bilateralSegment.h"
#include "DispProcessor.h"
#include "stereoMatch.h"
#include <ctime>
#include <string>
#include <stdlib.h>

using namespace std;
using namespace cv;

void runWithDisp()
{
	cout << "start---------" << endl << endl << endl;
	const char *leftImgDir = "data\\left_image_folder\\";
	const char *dispImgDir = "data\\disparity_image_folder\\";
	const char *resultImgDir = "data\\result_image_folder\\";
	int totalFrames = 40;
	int strideOfImgIdx = 5;
	Size frameSize = imread("data\\left_image_folder\\0.png", 0).size();
	//loop for all frames
	for (int startInd = 0; startInd < totalFrames; startInd += maxFrame)
	{
		int endInd = startInd + maxFrame - 1 < totalFrames - 1 ? startInd + maxFrame - 1 : totalFrames - 1;
		cout << "From " << startInd << " to " << endInd << endl << endl;
		// load video	
		vector<Mat> imgs;
		vector<Mat> frames;
		vector<Mat> disps;
		vector<Rect> rects;			
		int rect_area;
		Mat startMask;
		Mat startBilateralData;
		Mat result(frameSize.height, frameSize.width * 3, CV_8UC3);

		//loop in a block
		for (int i = startInd; i <= endInd; ++i)
		{			
			char disp_filename[100], img_filename[100];
			if (i < 10)
			{
				sprintf(img_filename, "data\\left_image_folder\\%d.png", strideOfImgIdx*i);
				sprintf(disp_filename, "data\\disparity_image_folder\\%d.png", strideOfImgIdx*i);
			}
			else
			{
				sprintf(img_filename, "data\\left_image_folder\\%d.png", strideOfImgIdx*i);
				sprintf(disp_filename, "data\\disparity_image_folder\\%d.png", strideOfImgIdx*i);
			}
		
			Mat img = imread(img_filename);
			Mat disp = imread(disp_filename, 0);
			if (!img.data || !disp.data) {
				cout << "invalid left image or disparity image file" << endl;
				system("pause");
				return;
			}
			frames.push_back(img);
			Mat contour;
			Rect rect(0, 0, frameSize.width, frameSize.height);
			//calculate disparity mask and maxDisp of foreground
			int maxDisp = calcContourFromDepth(disp, contour);
			//find out the valid area
			calcRect(img, contour, rect);
			imshow("valid area", contour);
			waitKey(1);
			Mat disp_roi = disp(rect);

			//remove pixels whose disparity is greater than maxDisp
			for (int y = 0; y < disp_roi.rows; ++y)
			{
				uchar *disp_roi_data = disp_roi.ptr<uchar>(y);
				for (int x = 0; x < disp_roi.cols; ++x)
				{
					if (disp_roi_data[x] > maxDisp)
						disp_roi_data[x] = 0;
				}
			}

			imgs.push_back(img(rect));
			disps.push_back(disp_roi);
			rects.push_back(rect);

			//for the first frame, use dispariy based algorithm
			if (i == startInd)
			{		
				cout << "start Disp + Bilateral Grid..." << endl;
				double startTic = getTickCount();
				rect_area = rect.area();
				liftMulFrame(imgs, disps, rects, startBilateralData);
				Mat startBilateralValues = Mat::ones(startBilateralData.rows, 1, CV_8UC1);
				Mat startSplattedData, startSplattedDisp;
				splat(startBilateralData, startBilateralValues, startSplattedData);
				Mat_<double> maskValues(startBilateralData.rows, 2);
				Mat dispBilaData = startBilateralData.col(DIM_OF_FEATURE - 2);
				for (int y = 0; y < maskValues.rows; ++y)
				{
					maskValues[y][0] = (dispBilaData.at<double>(y, 0) - 1.0) / (GRIDSIZE::dispGridSize - 1);
					maskValues[y][1] = 1 - (dispBilaData.at<double>(y, 0) - 1.0) / (GRIDSIZE::dispGridSize - 1);
				}
				splat(startBilateralData, maskValues, startSplattedDisp);
				maskValues.release();
				int *labels = graphcut(startSplattedData, startSplattedDisp);
				Mat labels_mat(prod(gridSize, DIM_OF_FEATURE), 1, CV_32SC1, labels);
				startSplattedData.release();
				startSplattedDisp.release();			
				Mat startSliced;
				sliceLabels(labels_mat, startBilateralData, startSliced);
				double *ls = startSliced.ptr<double>(0);
				int startCount = 0;
				startMask = Mat::zeros(imgs[0].size(), CV_8UC1);
				//if ls >0.5, than label =1 else label =0
				for (int y = 0; y < imgs[0].rows; ++y)
				{
					uchar *seg_mask_data = startMask.ptr<uchar>(y);
					for (int x = 0; x < imgs[0].cols; ++x)
					{
						int totalInd = startCount + x*imgs[0].rows + y;
						double temp = 255 * ls[totalInd];
						if (temp > 128)
						{
							seg_mask_data[x] = 255;
						}
					}
				}
				double startToc = getTickCount();
				cout << "Disp + Bilateral Grid takes " << (startToc - startTic) / getTickFrequency() << endl;
				startCount += imgs[0].rows*imgs[0].cols;

				//post processing
				medianBlur(startMask, startMask, 3);
				removeSmallComponent(startMask, 255, 100);
				removeSmallComponentForContour(startMask);
			}
		}

		//for other frames in a block, use initial mask method
		cout << "start Initial Mask + Bilateral Grid..." << endl;
		cout << '\t' << "start lifting... " << endl;
		Mat bilateralData;
		double tic = getTickCount();
		liftMulFrame(imgs, disps, rects, bilateralData);
		double toc1 = getTickCount();
		cout << '\t' <<"lifting " << maxFrame << " frames takes " << (toc1 - tic) / getTickFrequency() << " s" << endl;
		cout << '\t'  << "start splatting..." << endl;
		Mat bilateralMask;
		liftMask(imgs[0], disps[0], bilateralMask);
		startBilateralData = bilateralMask;
		Mat maskValues;
		binMask(startMask, maskValues);
		Mat bilateralValues = Mat::ones(bilateralData.rows, 1, CV_8UC1);
		Mat splattedData, splattedMask;
		splat(bilateralData, bilateralValues, splattedData);
		splat(startBilateralData, maskValues, splattedMask);
		maskValues.release();
		startBilateralData.release();
		double toc2 = getTickCount();
		cout << '\t' << "splatting " << maxFrame << " frames takes " << (toc2 - toc1) / getTickFrequency() << " s" << endl;

		cout << '\t' << "start graphCut..." << endl;
		int *labels = graphcut(splattedData, splattedMask, true);
		Mat labels_mat(prod(gridSize, DIM_OF_FEATURE), 1, CV_32SC1, labels);
		double toc3 = getTickCount();
		cout << '\t' << "graphCut " << " takes " << (toc3 - toc2) / getTickFrequency() << " s" << endl;
		splattedData.release();
		splattedMask.release();

		cout << '\t' << "start slicing..." << endl;
		Mat sliced;
		sliceLabels(labels_mat, bilateralData, sliced);
		bilateralData.release();
		double *ls = sliced.ptr<double>(0);

		double toc4 = getTickCount();
		cout << '\t' << "slicing " << maxFrame << " frames takes " << (toc4 - toc3) / getTickFrequency() << " s" << endl << endl;
		cout << "Initial Mask + Bilateral Grid takes " << (toc4 - tic) / getTickFrequency()/totalFrames << endl;
		int startCount = 0;
		for (int i = 0; i < maxFrame; ++i)
		{
			Mat seg_mask = Mat::zeros(frameSize, CV_8UC1);
			Mat seg_rgb = Mat::zeros(frameSize, CV_8UC3);
			for (int y = 0; y < imgs[i].rows; ++y)
			{
				uchar *seg_mask_data = seg_mask.ptr<uchar>(y + rects[i].y);
				for (int x = 0; x < imgs[i].cols; ++x)
				{
					int totalInd = startCount + x*imgs[i].rows + y;
					double temp = 255 * ls[totalInd];
					if (temp > 128)
					{
						seg_mask_data[x + rects[i].x] = 255;
					}
				}
			}

			startCount += imgs[i].rows*imgs[i].cols;
			//post processing
			medianBlur(seg_mask, seg_mask, 3);
			removeSmallComponent(seg_mask, 255, 100);
			removeSmallComponentForContour(seg_mask);
			//get segment image and save results
			for (int y = 0; y < imgs[i].rows; ++y)
			{
				uchar *seg_mask_data = seg_mask.ptr<uchar>(y + rects[i].y);
				Vec3b *seg_rgb_data = seg_rgb.ptr<Vec3b>(y + rects[i].y);
				Vec3b *rgb_data = imgs[i].ptr<Vec3b>(y);
				for (int x = 0; x < imgs[i].cols; ++x)
				{
					if (seg_mask_data[x + rects[i].x] == 255)
					{
						seg_rgb_data[x + rects[i].x][0] = rgb_data[x][0];
						seg_rgb_data[x + rects[i].x][1] = rgb_data[x][1];
						seg_rgb_data[x + rects[i].x][2] = rgb_data[x][2];
					}

				}
			}
			Mat seg_mask_c3;
			cvtColor(seg_mask, seg_mask_c3, CV_GRAY2BGR);
			frames[i].copyTo(result(Range::all(), Range(0, frameSize.width)));
			seg_mask_c3.copyTo(result(Range::all(), Range(frameSize.width, 2 * frameSize.width)));
			seg_rgb.copyTo(result(Range::all(), Range(2 * frameSize.width, 3 * frameSize.width)));
			char s1[100], s2[100];
			int ii = (i + startInd);
				char s3[100];
				if (ii  < 10)
					sprintf(s3, "data\\result_image_folder\\00%d.png", ii);
				else
					sprintf(s3, "data\\result_image_folder\\0%d.png", ii);
				imwrite(s3, seg_mask);
			imshow("result", result);
			waitKey(1);
		}
	}	
	cout << "-----------------------end!-----------------" << endl;
}

int main()
{
	runWithDisp();
	return 1;
}