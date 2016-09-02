// HistogramPlot.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

enum Mask { NOMASK, MASK, MASKINVERTED };

void normalize2(const cv::Mat &hist, cv::Mat &hist_norm, const double max) {
	hist_norm = hist / max;
}

int main()
{
	const std::string grca = "000298";

	const std::string img_path = "..\\..\\Dataset\\grca_RGB_" + grca + ".bmp";
	const std::string mask_path = "..\\..\\Dataset\\grca_RGB_" + grca + "_gt.bmp";
 

	const Mask maskmode = Mask::NOMASK;
	//const Mask maskmode = Mask::MASK;
	//const Mask maskmode = Mask::MASKINVERTED;

	cv::Mat src, mask, dst;

	/// Load image
	src = cv::imread(img_path, cv::IMREAD_COLOR);
	if (!src.data) { std::cout << "Could not open or find the image" << std::endl; return -1; }

	/// Load mask image
	mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
	if (!mask.data) { std::cout << "Could not open or find the mask image" << std::endl; return -1; }

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", src);

	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	/*
	&bgr_planes[0]: The source array(s)
	1: The number of source arrays (in this case we are using 1. We can enter here also a list of arrays )
	0: The channel (dim) to be measured. In this case it is just the intensity (each array is single-channel) so we just write 0.
	Mat(): A mask to be used on the source array ( zeros indicating pixels to be ignored ). If not defined it is not used
	b_hist: The Mat object where the histogram will be stored
	1: The histogram dimensionality.
	histSize: The number of bins per each used dimension
	histRange: The range of values to be measured per each dimension
	uniform and accumulate: The bin sizes are the same and the histogram is cleared at the beginning.
	*/

	switch (maskmode) {
	case Mask::NOMASK: mask = cv::Mat(); break;
	case Mask::MASKINVERTED: bitwise_not(mask, mask); break;
	}

	cv::calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);


	/// Normalize the result to [ 0, 1 ]
	cv::Mat b_dummy, g_dummy, r_dummy;
	cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_dummy, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_dummy, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_dummy, 1, &histSize, &histRange, uniform, accumulate);
	double minr, ming, minb, maxr, maxg, maxb, max;
	cv::minMaxIdx(b_dummy, &minb, &maxb);
	cv::minMaxIdx(g_dummy, &ming, &maxg);
	cv::minMaxIdx(r_dummy, &minr, &maxr);
	max = std::max({ maxr, maxg, maxb });


	cv::Mat b_hist_norm, g_hist_norm, r_hist_norm;
	//cv::normalize(b_hist, b_hist_norm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	//cv::normalize(g_hist, g_hist_norm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	//cv::normalize(r_hist, r_hist_norm, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	normalize2(b_hist, b_hist_norm, max);
	normalize2(g_hist, g_hist_norm, max);
	normalize2(r_hist, r_hist_norm, max);

	//#########################################################################
	//#########################################################################
	//#########################################################################

	histplot(256, r_hist_norm, g_hist_norm, b_hist_norm);


	//#########################################################################
	cv::waitKey(0);
}

