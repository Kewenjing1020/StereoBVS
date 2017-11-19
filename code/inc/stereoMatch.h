#include "common.h"

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

static void stereoMatch(Mat &img1, Mat &img2, Mat &disp8)
{
	const char* intrinsic_filename = 0;
	const char* extrinsic_filename = 0;
	const char* disparity_filename = 0;
	const char* point_cloud_filename = 0;

	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3 };
	int alg = STEREO_SGBM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	bool no_display = true;
	float scale = 1.f;

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	int color_mode = alg == STEREO_BM ? 0 : -1;

	if (scale != 1.f)
	{
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}

	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;

	if (intrinsic_filename)
	{
		// reading intrinsic parameters
		Mat M1, D1, M2, D2;
		M1 *= scale;
		M2 *= scale;
		Mat R, T, R1, P1, R2, P2;
		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);
		img1 = img1r;
		img2 = img2r;
	}

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = img1.channels();

	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = alg == STEREO_HH;

	var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
	var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
	var.nIt = 25;
	var.minDisp = -numberOfDisparities;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
	var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING;

	Mat disp;
	int64 t = getTickCount();
	if (alg == STEREO_BM)
		bm(img1, img2, disp);
	else if (alg == STEREO_VAR) {
		var(img1, img2, disp);
	}
	else if (alg == STEREO_SGBM || alg == STEREO_HH)
		sgbm(img1, img2, disp);
	t = getTickCount() - t;

	if (alg != STEREO_VAR)
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	else
		disp.convertTo(disp8, CV_8U);
	if (!no_display)
	{
		namedWindow("left", 1);
		imshow("left", img1);
		namedWindow("right", 1);
		imshow("right", img2);
		namedWindow("disparity", 0);
		imshow("disparity", disp8);
		printf("press any key to continue...");
		fflush(stdout);
		waitKey();
		printf("\n");
	}

	if (disparity_filename)
		imwrite(disparity_filename, disp8);

	if (point_cloud_filename)
	{
		printf("storing the point cloud...");
		fflush(stdout);
		Mat xyz;
		reprojectImageTo3D(disp, xyz, Q, true);
		saveXYZ(point_cloud_filename, xyz);
		printf("\n");
	}
}
