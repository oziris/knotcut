// Knotcut.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "segmentation.h"

#define FG_SEED 255
#define BG_SEED 0

bool running = true;

void print_img_size(const cv::Mat& image, const std::string& info = "")
{
	if (info == "")
		std::cout << "[ Rows: " << image.rows << ", Cols: " << image.cols << " ]" << std::endl;
	else
		std::cout << info << ": [ Rows: " << image.rows << ", Cols: " << image.cols << " ]" << std::endl;
}

void print_timing(const std::string info, const int64 before, const int64 after)
{
	double duration = ((double)(after - before)) / cv::getTickFrequency();
	if (duration < 1)
		std::cout << "[" << info << "] Time: " << duration * 1000 << " ms" << std::endl;
	else
		std::cout << "[" << info << "] Time: " << duration << " s" << std::endl;
}

void print_timing(const std::string info, const double duration)
{
	if (duration < 1)
		std::cout << "[" << info << "] Time: " << duration * 1000 << " ms" << std::endl;
	else
		std::cout << "[" << info << "] Time: " << duration << " s" << std::endl;
}

cv::Mat create_samples_vector(const cv::Mat& image, const cv::Mat& mask, const unsigned int value)
{
	if (image.rows != mask.rows || image.cols != mask.cols) return cv::Mat();

	std::vector<cv::Vec3b> tmp;
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (mask.at<uchar>(y, x) == value) {
				tmp.push_back(image.at<cv::Vec3b>(y, x));
			}
		}
	}

	cv::Mat reshaped_image, samples;
	cv::Mat tmp_image(tmp, true);

	reshaped_image = tmp_image.reshape(1, tmp.size());
	reshaped_image.convertTo(samples, CV_64FC1, 1.0 / 255.0);

	return samples;
}

void show_progress()
{
	while (running)
	{
		std::cout << ".";
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	std::cout << " DONE!" << std::endl;
}

void learn_em_model(cv::Ptr<cv::ml::EM> *model, cv::Mat *samples, cv::Mat *labels, cv::Mat *probs, cv::Mat *log_likelihoods, double *time)
{
	int64 before, after;
	// =================> START TIMING <========================
	before = cv::getTickCount();
	// =========================================================

	// execute EM Algorithm
	(*model)->trainEM(*samples, *log_likelihoods, *labels, *probs);

	// =================>  END  TIMING <========================
	after = cv::getTickCount();
	*time = ((double)(after - before)) / cv::getTickFrequency();
	// =========================================================
}

int main()
{
	const std::string grca = "000298";

	const std::string img_path = "..\\..\\Dataset\\grca_RGB_" + grca + ".bmp";
	const std::string seed_path = "..\\..\\Dataset\\grca_RGB_" + grca + "_seed.bmp";

	if (!boost::filesystem::exists(img_path)) { std::cout << "Error: image file does not exists\n"; return -1; }
	if (!boost::filesystem::exists(seed_path)) { std::cout << "Error: seed image file does not exists\n"; return -1; }

	cv::Mat src_img = cv::imread(img_path);
	cv::Mat seed_img = cv::imread(seed_path, cv::IMREAD_GRAYSCALE);

	if (!src_img.data) { std::cout << "Could not open the image." << std::endl; return -1; }
	if (!seed_img.data) { std::cout << "Could not open the seed image." << std::endl; return -1; }

	if (src_img.rows != seed_img.rows || src_img.cols != seed_img.cols)
	{
		std::cout << "The image and the seed mask image dimensions don't match." << std::endl;
		print_img_size(src_img, "Image");
		print_img_size(seed_img, "Seed");
		return -1;
	}

	print_img_size(src_img);
	const unsigned int image_rows = src_img.rows;
	const unsigned int image_cols = src_img.cols;

	cv::Mat fg_samples = create_samples_vector(src_img, seed_img, FG_SEED);
	cv::Mat bg_samples = create_samples_vector(src_img, seed_img, BG_SEED);

	std::cout << "FG: number of samples = " << fg_samples.rows << ", dims = " << fg_samples.cols << std::endl;
	std::cout << "BG: number of samples = " << bg_samples.rows << ", dims = " << bg_samples.cols << std::endl;
	std::cout << std::endl;


	// EM parameters
	int cluster_num{ 5 };

	cv::Ptr<cv::ml::EM> fg_em_model = cv::ml::EM::create();
	cv::Ptr<cv::ml::EM> bg_em_model = cv::ml::EM::create();

	// Set parameter
	fg_em_model->setClustersNumber(cluster_num);
	fg_em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
	fg_em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
	cv::ml::EM::DEFAULT_MAX_ITERS, 1e-6));

	bg_em_model->setClustersNumber(cluster_num);
	bg_em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
	bg_em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
	cv::ml::EM::DEFAULT_MAX_ITERS, 1e-6));

	
	// Prepare outputs
	cv::Mat fg_labels, bg_labels;
	cv::Mat fg_probs, bg_probs;
	cv::Mat fg_log_likelihoods, bg_log_likelihoods;
	double fg_time = 0.0, bg_time = 0.0;

	std::thread fg_thread(learn_em_model, &fg_em_model, &fg_samples, &fg_labels, &fg_probs, &fg_log_likelihoods, &fg_time);
	std::cout << "FG EM learning started." << std::endl;
	std::thread bg_thread(learn_em_model, &bg_em_model, &bg_samples, &bg_labels, &bg_probs, &bg_log_likelihoods, &bg_time);
	std::cout << "BG EM learning started." << std::endl;
	std::thread progress_thread(show_progress);

	// Synchronize threads:
	fg_thread.join();
	bg_thread.join();
	running = false;
	progress_thread.join();
	print_timing("FG EM", fg_time);
	print_timing("BG EM", bg_time);	
	

	Segmentation S;

	S.set_em_models(fg_em_model, bg_em_model);

	cv::Mat seed_img_inv = 255 - seed_img;
	S.set_images(src_img, false, cv::Mat(), true, seed_img_inv);
	

	S.init_segmentation();

	cv::Mat foreground_image;
	cv::Mat background_image;
	cv::Mat t_links_image;
	cv::Mat n_links_image;
	cv::Mat foreground_mask;
	cv::Mat background_mask;

	S.get_images(foreground_image, background_image);
	S.get_aux_images(t_links_image, n_links_image);
	S.get_aux_masks(foreground_mask, background_mask);

	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image", src_img);
	cv::namedWindow("Foreground", cv::WINDOW_AUTOSIZE);
	cv::imshow("Foreground", foreground_image);
	cv::namedWindow("Background", cv::WINDOW_AUTOSIZE);
	cv::imshow("Background", background_image);
	cv::namedWindow("T-links", cv::WINDOW_AUTOSIZE);
	cv::imshow("T-links", t_links_image);
	cv::namedWindow("N-links", cv::WINDOW_AUTOSIZE);
	cv::imshow("N-links", n_links_image);
	cv::namedWindow("Foreground mask", cv::WINDOW_AUTOSIZE);
	cv::imshow("Foreground mask", foreground_mask);
	cv::namedWindow("Background mask", cv::WINDOW_AUTOSIZE);
	cv::imshow("Background mask", background_mask);

	
	cv::waitKey(0);

    return 0;
}

