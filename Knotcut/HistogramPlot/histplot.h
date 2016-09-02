#include "stdafx.h"

PyObject *makelist(const cv::Mat *hist, const size_t size);

void histplot(const long N, const cv::Mat &hist_red, const cv::Mat &hist_green, const cv::Mat &hist_blue);