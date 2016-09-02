#include "stdafx.h"

#include "segmentation.h"


Segmentation::Segmentation()
{
	number_of_neigbours_ = 8;
	number_of_clusters_ = 5;

	flow_ = 0.0;

	fg_seed_on_ = false;
	bg_seed_on_ = false;

	for (unsigned int i = 0; i < number_of_neigbours_ / 2; i++)
		n_links_.push_back(cv::Mat());
}

bool Segmentation::load_em_models(const std::string& fg_filename, const std::string bg_filename) 
{
	try
	{
		fg_em_model_ = cv::ml::StatModel::load<cv::ml::EM>(fg_filename);
		bg_em_model_ = cv::ml::StatModel::load<cv::ml::EM>(bg_filename);

		fg_means_ = fg_em_model_->getMeans();
		bg_means_ = bg_em_model_->getMeans();
		fg_weights_ = fg_em_model_->getWeights();
		bg_weights_ = bg_em_model_->getWeights();
		fg_em_model_->getCovs(fg_covs_);
		bg_em_model_->getCovs(bg_covs_);
	}
	catch (...)
	{
		std::cout << "Crash in read method." << std::endl;
		return false;
	}
	return true;
}

bool Segmentation::set_em_models(const cv::Ptr<cv::ml::EM> fg_em_model, const cv::Ptr<cv::ml::EM> bg_em_model)
{
	try
	{
		fg_em_model_ = fg_em_model;
		bg_em_model_ = bg_em_model;

		fg_means_ = fg_em_model_->getMeans();
		bg_means_ = bg_em_model_->getMeans();
		fg_weights_ = fg_em_model_->getWeights();
		bg_weights_ = bg_em_model_->getWeights();
		fg_em_model_->getCovs(fg_covs_);
		bg_em_model_->getCovs(bg_covs_);
	}
	catch (...)
	{
		std::cout << "Crash in read method." << std::endl;
		return false;
	}
	return true;
}

bool Segmentation::load_images(const std::string& image_filename, const std::string& fg_mask_filename, const std::string& bg_mask_filename)
{
	src_img_ = cv::imread(image_filename);
	if (!src_img_.data)
	{
		std::cout << "Could not open or find the image: " << image_filename << std::endl;
		return false;
	}

	//int sigma = 2;
	//int ksize = (sigma * 5) | 1;
	//cv::GaussianBlur(src_img_, src_img_, cv::Size(ksize, ksize), sigma, sigma);

	rows_ = src_img_.rows; 
	cols_ = src_img_.cols;
	src_img_.convertTo(src_img_real_, CV_64FC1, 1.0 / 255.0);


	if (fg_mask_filename != "")
	{
		fg_seed_img_ = cv::imread(fg_mask_filename, cv::IMREAD_GRAYSCALE);
		if (!fg_seed_img_.data)
		{
			std::cout << "Could not open or find the image: " << fg_mask_filename << std::endl;
			return false;
		}
		fg_seed_on_ = true;
	}

	if (bg_mask_filename != "")
	{
		bg_seed_img_ = cv::imread(bg_mask_filename, cv::IMREAD_GRAYSCALE);
		if (!bg_seed_img_.data)
		{
			std::cout << "Could not open or find the image: " << bg_mask_filename << std::endl;
			return false;
		}
		bg_seed_on_ = true;
	}

	return true;
}

bool Segmentation::set_images(const cv::Mat& image, const bool fg_seed_on, const cv::Mat& fg_seed, const bool bg_seed_on, const cv::Mat& bg_seed)
{
	src_img_ = image;


	if (!src_img_.data) { std::cout << "Fatal error: set source image." << std::endl; return false; }

	//int sigma = 2;
	//int ksize = (sigma * 5) | 1;
	//cv::GaussianBlur(src_img_, src_img_, cv::Size(ksize, ksize), sigma, sigma);

	rows_ = src_img_.rows;
	cols_ = src_img_.cols;
	src_img_.convertTo(src_img_real_, CV_64FC1, 1.0 / 255.0);


	if (fg_seed_on)
	{
		fg_seed_img_ = fg_seed;
		if (!fg_seed_img_.data) { std::cout << "Fatal error: set foreground seed image." << std::endl; return false; }
		fg_seed_on_ = true;
	}

	if (bg_seed_on)
	{
		bg_seed_img_ = bg_seed;
		if (!bg_seed_img_.data) { std::cout << "Fatal error: set background seed image." <<  std::endl; return false; }
		bg_seed_on_ = true;
	}

	return true;
}

void Segmentation::create_trimap()
{
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			if (fg_seed_on_ && fg_seed_img_.at<uchar>(y, x) == mask_value_)
				trimap_img_.at<uchar>(y, x) = Trimap::FOREGROUND;

			if (bg_seed_on_ && bg_seed_img_.at<uchar>(y, x) == mask_value_)
				trimap_img_.at<uchar>(y, x) = Trimap::BACKGROUND;
		}
	}
}

double Segmentation::distance(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
{
	return sqrt((double)((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1)));
}

double Segmentation::distance2(const cv::Vec3d& c0, const cv::Vec3d& c1)
{
	cv::Vec3d diff = c0 - c1;
	double dist2 = diff.dot(diff);
	return dist2;
}

double Segmentation::compute_beta()
{
	if (!src_img_.data)
	{
		std::cout << "Source image not present." << std::endl;
		return 0.0;
	}

	double tmp_sum = 0.0;
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			cv::Vec3d color1 = src_img_real_.at<cv::Vec3d>(y, x);
			if (x > 0) // left
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y, x - 1);
				tmp_sum += distance2(color1, color2);
			}
			if (y > 0 && x > 0) // upleft
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x - 1);
				tmp_sum += distance2(color1, color2);
			}
			if (y > 0) // up
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x);
				tmp_sum += distance2(color1, color2);
			}
			if (y > 0 && x < cols_ - 1) // upright
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x + 1);
				tmp_sum += distance2(color1, color2);
			}
		}
	}

	double tmp = 0.0;
	if (tmp_sum <= std::numeric_limits<double>::epsilon())
		tmp = 0.0;
	else
		tmp = 1.0 / (2 * tmp_sum / (4 * cols_ * rows_ - 3 * cols_ - 3 * rows_ + 2));

	return tmp;
}

double Segmentation::compute_lambda()
{
	if (!src_img_.data)
	{
		std::cout << "Source image not present." << std::endl;
		return 0.0;
	}

	return 8 * gamma_ + 1;
}

void Segmentation::compute_N_links()
{
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			cv::Vec3d color1 = src_img_real_.at<cv::Vec3d>(y, x);
			if (x > 0) // left
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y, x - 1);
				n_links_[Direction::LEFT].at<double>(y, x) = gamma_ * exp(-beta_ * distance2(color1, color2)) / distance(x, y, x - 1, y);
			}
			else
			{
				n_links_[Direction::LEFT].at<double>(y, x) = 0.0;
			}
			if (y > 0 && x > 0) // upleft
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x - 1);
				n_links_[Direction::UP_LEFT].at<double>(y, x) = gamma_ * exp(-beta_ * distance2(color1, color2)) / distance(x, y, x - 1, y - 1);
			}
			else
			{
				n_links_[Direction::UP_LEFT].at<double>(y, x) = 0.0;
			}
			if (y > 0) // up
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x);
				n_links_[Direction::UP].at<double>(y, x) = gamma_ * exp(-beta_ * distance2(color1, color2)) / distance(x, y, x, y - 1);
			}
			else
			{
				n_links_[Direction::UP].at<double>(y, x) = 0.0;
			}
			if (y > 0 && x < cols_ - 1) // upright
			{
				cv::Vec3d color2 = src_img_real_.at<cv::Vec3d>(y - 1, x + 1);
				n_links_[Direction::UP_RIGHT].at<double>(y, x) = gamma_ * exp(-beta_ * distance2(color1, color2)) / distance(x, y, x + 1, y - 1);
			}
			else
			{
				n_links_[Direction::UP_RIGHT].at<double>(y, x) = 0.0;
			}
		}
	}
}

cv::Mat Segmentation::invert_3x3_matrix(const cv::Mat& m)
{
	/*
	[ a b c
	d e f
	g h i ]

	A = (ei-fh)   D = -(bi-ch)  G = (bf-ce)
	B = -(di-fg)  E = (ai-cg)   H = -(af-cd)
	C = (dh-eg)   F = -(ah-bg)  I = (ae-bd)

	[ A D G
  	B E H
	C F I ]
	*/

	cv::Mat m_ = (cv::Mat_<double>(3, 3) <<
		m.at<double>(1, 1) * m.at<double>(2, 2) - m.at<double>(1, 2) * m.at<double>(2, 1),		// A = (ei - fh)   
		-(m.at<double>(0, 1) * m.at<double>(2, 2) - m.at<double>(0, 2) * m.at<double>(2, 1)),	// D = -(bi - ch)  
		m.at<double>(0, 1) * m.at<double>(1, 2) - m.at<double>(0, 2) * m.at<double>(1, 1),		// G = (bf - ce)
		-(m.at<double>(1, 0) * m.at<double>(2, 2) - m.at<double>(1, 2) * m.at<double>(2, 0)),	// B = -(di - fg)  
		m.at<double>(0, 0) * m.at<double>(2, 2) - m.at<double>(0, 2) * m.at<double>(2, 0),		// E = (ai - cg)   
		-(m.at<double>(0, 0) * m.at<double>(1, 2) - m.at<double>(0, 2) * m.at<double>(1, 0)),	// H = -(af - cd)
		m.at<double>(1, 0) * m.at<double>(2, 1) - m.at<double>(1, 1) * m.at<double>(2, 0),		// C = (dh - eg)   
		-(m.at<double>(0, 0) * m.at<double>(2, 1) - m.at<double>(0, 1) * m.at<double>(2, 0)),	// F = -(ah - bg)  
		m.at<double>(0, 0) * m.at<double>(1, 1) - m.at<double>(0, 1) * m.at<double>(1, 0)		// I = (ae - bd)
		);

	return m_ / cv::determinant(m);
}

void Segmentation::compute_dets_and_inv_covs()
{
	for (auto& cov : fg_covs_)
	{
		fg_dets_.push_back(cv::determinant(cov));
		fg_inv_covs_.push_back(invert_3x3_matrix(cov));
	}

	for (auto& cov : bg_covs_)
	{
		bg_dets_.push_back(cv::determinant(cov));
		bg_inv_covs_.push_back(invert_3x3_matrix(cov));
	}
}

double Segmentation::compute_prob(const cv::Vec3d& color, const cv::Mat& means, const cv::Mat& weights, const std::vector<double>& dets, const std::vector<cv::Mat>& inv_covs)
{
	double p = 0.0;

	for (unsigned int i = 0; i < number_of_clusters_; i++)
	{
		double pi = weights.at<double>(0, i);
		double det = dets[i];
		cv::Vec3d mu = means.at<cv::Vec3d>(i, 0);
		cv::Mat inv_cov = inv_covs[i];

		cv::Vec3d diff = color - mu;
		
		double tmp = 
			  (diff[0] * inv_cov.at<double>(0, 0) + diff[1] * inv_cov.at<double>(1, 0) + diff[2] * inv_cov.at<double>(2, 0)) * diff[0] 
			+ (diff[0] * inv_cov.at<double>(0, 1) + diff[1] * inv_cov.at<double>(1, 1) + diff[2] * inv_cov.at<double>(2, 1)) * diff[1]
			+ (diff[0] * inv_cov.at<double>(0, 2) + diff[1] * inv_cov.at<double>(1, 2) + diff[2] * inv_cov.at<double>(2, 2)) * diff[2];
		
		p += pi / sqrt(det) * exp(-0.5 * tmp);
	}

	return p;
}

void Segmentation::compute_T_links()
{
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			cv::Vec3d color = src_img_real_.at<cv::Vec3d>(y, x);

			double fg_prob = compute_prob(color, fg_means_, fg_weights_, fg_dets_, fg_inv_covs_);
			double bg_prob = compute_prob(color, bg_means_, bg_weights_, bg_dets_, bg_inv_covs_);
			
			fg_prob_.at<double>(y, x) = fg_prob;
			bg_prob_.at<double>(y, x) = bg_prob;

			if (trimap_img_.at<uchar>(y, x) == Trimap::UNKNOWN)
			{
				double log_bg_prob = -log(bg_prob);
				double log_fg_prob = - log(fg_prob);

				fg_t_links_.at<double>(y, x) = log_bg_prob;
				bg_t_links_.at<double>(y, x) = log_fg_prob;

				int ddd = 0;
			}
			else if (trimap_img_.at<uchar>(y, x) == Trimap::BACKGROUND)
			{
				fg_t_links_.at<double>(y, x) = 0;
				bg_t_links_.at<double>(y, x) = lambda_;
			}
			else // if (trimap_img_.at<uchar>(y, x) = Trimap::FOREGROUND)
			{
				fg_t_links_.at<double>(y, x) = lambda_;
				bg_t_links_.at<double>(y, x) = 0;
			}
		}
	}
}

bool Segmentation::init_segmentation()
{
	if (fg_seed_on_ && (src_img_.rows != fg_seed_img_.rows || src_img_.cols != fg_seed_img_.cols))
	{
		std::cout << "The image and the foreground seed mask dimensions don't match." << std::endl;
		return false;
	}

	if (bg_seed_on_ && (src_img_.rows != bg_seed_img_.rows || src_img_.cols != bg_seed_img_.cols))
	{
		std::cout << "The image and the background seed mask dimensions don't match." << std::endl;
		return false;
	}

	fg_prob_ = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar(0));
	bg_prob_ = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar(0));

	fg_t_links_ = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar(0));
	bg_t_links_ = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar(0));

	n_links_img_ = cv::Mat(rows_, cols_, CV_8U, cv::Scalar(0));
	n_links_img_real_ = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar(0));
	
	t_links_img_ = cv::Mat(rows_, cols_, CV_8UC3, cv::Scalar(0,0,0));
	t_links_img_real_ = cv::Mat(rows_, cols_, CV_64FC3, cv::Scalar(0,0,0));

	trimap_img_ = cv::Mat(rows_, cols_, CV_8U, cv::Scalar::all(Trimap::UNKNOWN));
	segmentation_img_ = cv::Mat(rows_, cols_, CV_8U, cv::Scalar::all(Trimap::UNKNOWN));
	segmentation_fg_img_ = cv::Mat(rows_, cols_, CV_8UC3, cv::Scalar(0, 0, 0));
	segmentation_bg_img_ = cv::Mat(rows_, cols_, CV_8UC3, cv::Scalar(0, 0, 0));
	segmentation_fg_mask_ = cv::Mat(rows_, cols_, CV_8U, cv::Scalar(0));
	segmentation_bg_mask_ = cv::Mat(rows_, cols_, CV_8U, cv::Scalar(0));

	number_of_nodes_ = cols_ * rows_;
	number_of_edges_ = (4 * cols_ * rows_ - 3 * cols_ - 3 * rows_ + 2);

	create_trimap();

	beta_ = compute_beta();
	lambda_ = compute_lambda();

	for (auto& m : n_links_)
	{
		m = cv::Mat(rows_, cols_, CV_64FC1, cv::Scalar::all(0));
	}
	compute_N_links();

	compute_dets_and_inv_covs();
	compute_T_links();

	create_images();

	init_graph();

	if (graph_)
		flow_ = graph_->maxflow();

	process_segmentation();


	cv::Mat tmp1 = trimap_img_ * 100;
	cv::Mat tmp2 = segmentation_img_ * 100;

	return true;
}


void Segmentation::create_images()
{
	//double v_min = std::numeric_limits<double>::max();
	//double v_max = std::numeric_limits<double>::min();

	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			// N-links image 
			if (x > 0) // left
			{
				double weight = n_links_[Direction::LEFT].at<double>(y, x) / lambda_;
				n_links_img_real_.at<double>(y, x) += weight;
				n_links_img_real_.at<double>(y, x - 1) += weight;
				
				//if (n_links_img_real_.at<double>(y, x) < v_min) { v_min = n_links_img_real_.at<double>(y, x); }
				//if (n_links_img_real_.at<double>(y, x) > v_max) { v_max = n_links_img_real_.at<double>(y, x); }
			}
			if (y > 0 && x > 0) // upleft
			{
				double weight = n_links_[Direction::UP_LEFT].at<double>(y, x) / lambda_;
				n_links_img_real_.at<double>(y, x) += weight;
				n_links_img_real_.at<double>(y - 1, x - 1) += weight;

				//if (n_links_img_real_.at<double>(y, x) < v_min) { v_min = n_links_img_real_.at<double>(y, x); }
				//if (n_links_img_real_.at<double>(y, x) > v_max) { v_max = n_links_img_real_.at<double>(y, x); }
			}
			if (y > 0) // up
			{
				double weight = n_links_[Direction::UP].at<double>(y, x) / lambda_;
				n_links_img_real_.at<double>(y, x) += weight;
				n_links_img_real_.at<double>(y - 1, x) += weight;

				//if (n_links_img_real_.at<double>(y, x) < v_min) { v_min = n_links_img_real_.at<double>(y, x); }
				//if (n_links_img_real_.at<double>(y, x) > v_max) { v_max = n_links_img_real_.at<double>(y, x); }
			}
			if (y > 0 && x < cols_ - 1) // upright
			{
				double weight = n_links_[Direction::UP_RIGHT].at<double>(y, x) / lambda_;
				n_links_img_real_.at<double>(y, x) += weight;
				n_links_img_real_.at<double>(y - 1, x + 1) += weight;

				//if (n_links_img_real_.at<double>(y, x) < v_min) { v_min = n_links_img_real_.at<double>(y, x); }
				//if (n_links_img_real_.at<double>(y, x) > v_max) { v_max = n_links_img_real_.at<double>(y, x); }
			}

			// T-links image
			t_links_img_real_.at<cv::Vec3d>(y, x)[Color::R] = pow(fg_t_links_.at<double>(y, x) / lambda_, 0.25);
			t_links_img_real_.at<cv::Vec3d>(y, x)[Color::G] = pow(bg_t_links_.at<double>(y, x) / lambda_, 0.25);
		}
	}
}

void Segmentation::init_graph()
{
	// Set up the graph (it can only be used once, so we have to recreate it each time the graph is updated)
	if (graph_) delete graph_;

	graph_ = new GraphType(/*estimated # of nodes*/ number_of_nodes_, /*estimated # of edges*/ number_of_edges_);
	nodes_ = new GraphNodes(rows_, std::vector<GraphType::node_id>(cols_));

	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			(*nodes_)[y][x] = graph_->add_node();
		}
	}

	// Set T-Link weights
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			graph_->add_tweights((*nodes_)[y][x], fg_t_links_.at<double>(y, x), bg_t_links_.at<double>(y, x));
		}
	}

	// Set N-Link weights 
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{

			if (x > 0) 
			{
				graph_->add_edge((*nodes_)[y][x], (*nodes_)[y][x - 1], n_links_[Direction::LEFT].at<double>(y, x), n_links_[Direction::LEFT].at<double>(y, x));		
			}
			if (y > 0 && x > 0) 
			{
				graph_->add_edge((*nodes_)[y][x], (*nodes_)[y - 1][x - 1], n_links_[Direction::UP_LEFT].at<double>(y, x), n_links_[Direction::UP_LEFT].at<double>(y, x));
			}
			if (y > 0) 
			{
				graph_->add_edge((*nodes_)[y][x], (*nodes_)[y-1][x], n_links_[Direction::UP].at<double>(y, x), n_links_[Direction::UP].at<double>(y, x));
			}
			if (y > 0 && x < cols_ - 1) 
			{
				graph_->add_edge((*nodes_)[y][x], (*nodes_)[y-1][x + 1], n_links_[Direction::UP_RIGHT].at<double>(y, x), n_links_[Direction::UP_RIGHT].at<double>(y, x));	
			}
		}
	}
}

void Segmentation::process_segmentation()
{
	for (unsigned int y = 0; y < rows_; y++)
	{
		for (unsigned int x = 0; x < cols_; x++)
		{
			if (trimap_img_.at<uchar>(y, x) == Trimap::BACKGROUND)
			{
				segmentation_img_.at<uchar>(y, x) = Trimap::BACKGROUND;
				segmentation_bg_img_.at<cv::Vec3b>(y, x) = src_img_.at<cv::Vec3b>(y, x);
				segmentation_bg_mask_.at<uchar>(y, x) = 255;
			}
			else if (trimap_img_.at<uchar>(y, x) == Trimap::FOREGROUND)
			{
				segmentation_img_.at<uchar>(y, x) = Trimap::FOREGROUND;
				segmentation_fg_img_.at<cv::Vec3b>(y, x) = src_img_.at<cv::Vec3b>(y, x);
				segmentation_fg_mask_.at<uchar>(y, x) = 255;
			}
			else	
			{
				if (graph_->what_segment((*nodes_)[y][x]) == GraphType::SOURCE)
				{
					segmentation_img_.at<uchar>(y, x) = Trimap::FOREGROUND;
					segmentation_fg_img_.at<cv::Vec3b>(y, x) = src_img_.at<cv::Vec3b>(y, x);
					segmentation_fg_mask_.at<uchar>(y, x) = 255;
				}
				else
				{
					segmentation_img_.at<uchar>(y, x) = Trimap::BACKGROUND;
					segmentation_bg_img_.at<cv::Vec3b>(y, x) = src_img_.at<cv::Vec3b>(y, x);
					segmentation_bg_mask_.at<uchar>(y, x) = 255;
				}
			}
		}
	}
}


void Segmentation::get_images(cv::Mat& foreground, cv::Mat& background)
{
	foreground = segmentation_fg_img_;
	background = segmentation_bg_img_;
}

void Segmentation::get_aux_images(cv::Mat& t_links, cv::Mat& n_links)
{
	t_links = t_links_img_real_;
	n_links = n_links_img_real_;
}

void Segmentation::get_aux_masks(cv::Mat& foreground, cv::Mat& background)
{
	foreground = segmentation_fg_mask_;
	background = segmentation_bg_mask_;
}