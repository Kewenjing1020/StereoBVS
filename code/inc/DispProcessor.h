#pragma once
#include "common.h"
/*
	This header includes functions to process a disparity image
*/

void calculateGrayPeak(Mat &disparityMap, int &peak, int &loDiff, int &upDiff, int times) {
	int histSize = 255;
	Mat hist;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	calcHist(&disparityMap, 1, 0, Mat(), hist, 1, &histSize, &histRange);
	int total = disparityMap.rows * disparityMap.cols;
	peak = 0;
	vector<uchar> peaks;
	vector<uchar> troughs;
	float* _data_hist = (float*)hist.ptr(0);
	total = total - (int)_data_hist[0];

	for (int i = 250; i > 4; i--)
	{
		if (_data_hist[i] < total *0.008)
			continue;
		if (_data_hist[i] > _data_hist[i - 1] && _data_hist[i] > _data_hist[i + 1])
		{
			int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
				_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
			if (interval_total > total *0.03)
			{
				//if (times > 1)
				//{
				//	times--;
				//	continue;
				//}
				peak = i;
				break;
			}
		}
	}

	//for (int _upDiff = 0; ; _upDiff++)
	//{
	//	int index = peak + _upDiff;
	//	if (_data_hist[index] < 100 && _data_hist[index + 1] < 100 && _data_hist[index+2] < 100)
	//	{
	//		upDiff = _upDiff+2;
	//		break;
	//	}
	//}
	


	//find second peak.
	int second_peak = 0;
	bool concesstion = true;
	for (int i = peak - 4; i > 4; i--)
	{
		if (concesstion)
		{
			int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
				_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];

			if (interval_total > total / 10)
			{
				continue;
			}
			else
				concesstion = false;

		}
		else
		{
			if (_data_hist[i] > _data_hist[i - 1] && _data_hist[i] > _data_hist[i + 1])
			{
				if (_data_hist[i] < total / 100)
					continue;

				int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
					_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
				if (interval_total > total / 10)
				{
					second_peak = i;
					break;
				}
			}
		}
	}

	//if no second_peak
	
	for (loDiff = 0;loDiff < peak/2; loDiff++)
	{
		int i = peak - loDiff;
		int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
			_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
		if (_data_hist[i] < total/1000 && interval_total < total/100)
			break;
	}
	for (upDiff = 0; upDiff < peak / 2, upDiff + peak < 250; upDiff++)
	{
		int i = peak + upDiff;
		int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
			_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
		if (_data_hist[i] < total / 1000 && interval_total < total / 100)
			break;
	}
	//if (second_peak == 0)
	//{
	//	int one_peak_total = 0;
	//	for (int i = 0; i <= upDiff; i++)
	//	{
	//		one_peak_total += _data_hist[i + peak];
	//	}
	//	while (abs(one_peak_total) < 0.7 * total)
	//	{
	//		_loDiff++;
	//		one_peak_total += _data_hist[peak - _loDiff];
	//	}
	//	while (true)
	//	{
	//		_loDiff++;
	//		if (_data_hist[peak - _loDiff] < 100 && _data_hist[peak - _loDiff + 1] < 100 && _data_hist[peak - _loDiff + 2] < 100)
	//			break;
	//	}
	//	loDiff = _loDiff;
	//	return ;
	//}


	////if no second_peak
	//float _temp = _data_hist[peak];
	//for (int i = second_peak; i < peak; i++)
	//{
	//	if (_data_hist[i] < _temp)
	//	{
	//		_temp = _data_hist[i];
	//		_loDiff = i;
	//	}
	//}
	//loDiff = peak - _loDiff + 3;

}

void findSeedPoint(Mat &disparityMap, int &_x, int &_y, int peak) {
	uchar* _data_imageGray = disparityMap.ptr<uchar>(0);
	int total = disparityMap.cols * disparityMap.rows;

	RNG rng(getTickCount());
	for (;;) {
		int index = rng.uniform(0, total);
		if ((int)(_data_imageGray[index]) == peak)
		{
			_y = index / disparityMap.cols;
			_x = index % disparityMap.cols;
			break;
		}
	}
}


void removeSmallComponentForContour(Mat &binary_image) {
	Mat label_image;
	binary_image.convertTo(label_image, CV_32SC1);


	vector<int> labels;
	vector<Rect> rects;
	int max_rect_label = -1;
	Rect max_rect;
	int area_total = binary_image.rows * binary_image.cols;
	int label_count = 256;
	for (int _row = 0; _row < binary_image.rows; _row++)
	{
		int _index_row = _row;
		Rect _rect;
		int* _label_row_data = label_image.ptr<int>(_index_row);
		uchar* _binary_image_row_data = binary_image.ptr<uchar>(_index_row);
		for (int _col = 0; _col < binary_image.cols; _col++)
		{
			int _index_col = _col;
			if (_binary_image_row_data[_index_col] != 255)
			{
				continue;
			}
			if (_label_row_data[_index_col] > 255)
				continue;

			floodFill(label_image, cv::Point(_index_col, _index_row), label_count, &_rect, 0, 0, 8);

			if (_rect.area() > 0.25 * area_total)
			{
				max_rect = _rect;
				max_rect_label = label_count;
				label_count++;
				continue;
			}

			labels.push_back(label_count);
			rects.push_back(_rect);

			label_count++;
		}
	}

	if (max_rect_label == -1)
	{
		int area = 0;
		for (int i_rect = 0; i_rect < rects.size(); i_rect++)
		{
			if (rects[i_rect].area() > area)
			{
				area = rects[i_rect].area();
				max_rect_label = labels[i_rect];
				max_rect = rects[i_rect];
			}
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		if (max_rect_label == labels[i_rect])
		{
			rects.erase(rects.begin() + i_rect);
			labels.erase(labels.begin() + i_rect);
			break;
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		//Add more contraints.
		if (max_rect.contains(rects[i_rect].tl()) && max_rect.contains(rects[i_rect].br()))
		{
			rects.erase(rects.begin() + i_rect);
			labels.erase(labels.begin() + i_rect);
			i_rect--;
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		//cout << "rects: " << rects[i_rect] << endl;
		Rect _rect = rects[i_rect];
		int label = labels[i_rect];
		for (int _row = 0; _row < _rect.height; _row++) {
			int _index_row_2 = _rect.y + _row;
			int* _row_label_data = label_image.ptr<int>(_index_row_2);
			uchar* _binary_image_row_data = binary_image.ptr<uchar>(_index_row_2);
			for (int _col = 0; _col < _rect.width; _col++) {
				int _index_col_2 = _rect.x + _col;
				if (_index_col_2 >= label_image.cols)
					continue;
				if (_row_label_data[_index_col_2] != label) {
					continue;
				}
				_binary_image_row_data[_index_col_2] = 0;
			}
		}
	}

}

void removeSmallComponent(Mat &image, int removeColor, int diviseur) {
	int result = 0;
	if (removeColor == 0)
		result = 255;

	Mat label_image;
	image.convertTo(label_image, CV_32SC1);

	int area_total = image.rows * image.cols;
	int label_count = 256;
	for (int _row = 0; _row < image.rows; _row++)
	{
		int _index_row = _row;
		Rect _rect;
		int* _row_label_data = label_image.ptr<int>(_index_row);
		uchar* _row_contour_data = image.ptr<uchar>(_index_row);
		for (int _col = 0; _col < image.cols; _col++)
		{
			int _index_col = _col;
			if (_row_contour_data[_index_col] != removeColor)
			{
				continue;
			}
			if (_row_label_data[_index_col] > 255)
				continue;

			floodFill(label_image, cv::Point(_index_col, _index_row), label_count, &_rect, 0, 0, 4);

			if (_rect.area() < area_total / diviseur)
			{
				_row--;
				break;
			}

			label_count++;
		}

		for (int _row = 0; _row < _rect.height; _row++) {
			int _index_row_2 = _rect.y + _row;
			int* _row_label_data = label_image.ptr<int>(_index_row_2);
			for (int _col = 0; _col < _rect.width; _col++) {
				int _index_col_2 = _rect.x + _col;
				if (_row_label_data[_index_col_2] != label_count) {
					continue;
				}
				image.at<uchar>(_index_row_2, _index_col_2) = result;
			}
		}
	}
}

int calcContourFromDepth(Mat &disparityMap, Mat &contour_get_from_depth ) {
	int total = disparityMap.cols * disparityMap.rows;
	Rect rect;
	int label_count = 256;
	Mat label_image;
	int _x = -1, _y = -1;

	int times = 1;
	int peak, upDiff, loDiff;
	for (; rect.area() < total / 10;)
	{
		calculateGrayPeak(disparityMap, peak, loDiff, upDiff, times++);
		int have_tried = 0;
		do {
			have_tried++;
			findSeedPoint(disparityMap, _x, _y, peak);
			disparityMap.convertTo(label_image, CV_32SC1);
			floodFill(label_image, Point(_x, _y), label_count, &rect, loDiff, upDiff);
			if (have_tried > 20)
				break;
		} while (rect.area() < total / 10);
	}

	Mat _Contour = Mat(disparityMap.rows, disparityMap.cols, CV_8UC1, Scalar::all(0));
	for (int _row = 0; _row < rect.height; _row++)
	{
		int _index_row = rect.y + _row;
		int* _row_label_data = label_image.ptr<int>(_index_row);
		uchar* _row_disparity_data = disparityMap.ptr<uchar>(_index_row);
		uchar* _row_contour_data = _Contour.ptr<uchar>(_index_row);
		for (int _col = 0; _col < rect.width; _col++)
		{
			int _index_col = rect.x + _col;
			if (_row_label_data[_index_col] == label_count
				&&_row_disparity_data[_index_col] >= peak - loDiff
				&& _row_disparity_data[_index_col] <= peak + upDiff)
			{
				_row_contour_data[_index_col] = 255;
			}
		}
	}	
	removeSmallComponent(_Contour, 255, 100);
	removeSmallComponentForContour(_Contour);
	_Contour.copyTo(contour_get_from_depth);
	//_Contour.release();
	return peak + upDiff > 255 ? 255 : peak + upDiff;
}

void calcRect(Mat &img, Mat &contour, Rect &rect)
{
	int x0, x1, y0, y1;
	for (y0 = 0;; ++y0)
	{
		uchar *data = contour.ptr<uchar>(y0);
		bool isEnd = false;
		for (int x = 0; x < contour.cols; ++x)
		{
			if (data[x] == 255)
			{
				isEnd = true;
				break;
			}
		}
		if (isEnd)
			break;
	}
	for (y1 = contour.rows-1;; --y1)
	{
		uchar *data = contour.ptr<uchar>(y1);
		bool isEnd = false;
		for (int x = 0; x < contour.cols; ++x)
		{
			if (data[x] == 255)
			{
				isEnd = true;
				break;
			}
		}
		if (isEnd)
			break;
	}
	for (x0 = 0;; ++x0)
	{		
		bool isEnd = false;
		for (int y = y0; y < y1; ++y)
		{
			if (contour.at<uchar>(y,x0) == 255)
			{
				isEnd = true;
				break;
			}
		}
		if (isEnd)
			break;
	}

	for (x1 = contour.cols-1;; --x1)
	{
		bool isEnd = false;
		for (int y = y0; y < y1; ++y)
		{
			if (contour.at<uchar>(y, x1) == 255)
			{
				isEnd = true;
				break;
			}
		}
		if (isEnd)
			break;
	}
	int delta = 10;
	x0 = x0 - delta > 0 ? x0 - delta : 0;
	y0 = y0 - delta > 0 ? y0 - delta : 0;
	x1 = x1 + delta < contour.cols ? x1 + delta : contour.cols;
	y1 = y1 + delta < contour.rows ? y1 + delta : contour.rows;
	rect.x = x0;
	rect.y = y0;
	rect.height = ((y1 - y0) / 4) * 4;
	rect.width = ((x1 - x0) / 4) * 4;
	Mat contour_;
	contour.copyTo(contour_);
	cvtColor(contour_, contour_, CV_GRAY2BGR);
	line(contour_, Point(x0, y0), Point(x0, y1), Scalar(255, 0, 0), 3);
	line(contour_, Point(x0, y1), Point(x1, y1), Scalar(255, 0, 0), 3);
	line(contour_, Point(x1, y1), Point(x1, y0), Scalar(255, 0, 0), 3);
	line(contour_, Point(x1, y0), Point(x0, y0), Scalar(255, 0, 0), 3);
}