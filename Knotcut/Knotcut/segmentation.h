#include "stdafx.h"

#include "MaxFlow\graph.h"

class Segmentation
{
public:
	//typedef Graph<int, int, int> GraphType;
	typedef Graph<double, double, double> GraphType;
	typedef std::vector<std::vector<GraphType::node_id>> GraphNodes;

	enum Direction { LEFT, UP_LEFT, UP, UP_RIGHT };
	enum Trimap { UNKNOWN, BACKGROUND, FOREGROUND };
	enum Color { B, G, R};

	Segmentation();

	bool load_em_models(const std::string& fg_filename, const std::string bg_filename);
	//bool set_em_models(const cv::Ptr<cv::ml::EM> *fg_em_model, const cv::Ptr<cv::ml::EM> *bg_em_model);
	bool set_em_models(const cv::Ptr<cv::ml::EM> fg_em_model, const cv::Ptr<cv::ml::EM> bg_em_model);
	bool load_images(const std::string& image_filename, const std::string& fg_mask_filename = "", const std::string& bg_mask_filename = "");	
	bool set_images(const cv::Mat& image, const bool fg_seed_on, const cv::Mat& fg_seed, const bool bg_seed_on, const cv::Mat& bg_seed);
	bool init_segmentation();
	void get_images(cv::Mat& foreground, cv::Mat& background);
	void get_aux_images(cv::Mat& t_links, cv::Mat& n_links);
	void get_aux_masks(cv::Mat& foreground, cv::Mat& background);

private:
	const unsigned int mask_value_ = 255;
	const double gamma_ = 50;

	double beta_;
	double lambda_;
	double flow_;

	unsigned int number_of_neigbours_;
	unsigned int number_of_clusters_;
	unsigned int rows_, cols_;
	unsigned int number_of_nodes_;
	unsigned int number_of_edges_;

	cv::Ptr<cv::ml::EM> fg_em_model_;
	cv::Ptr<cv::ml::EM> bg_em_model_;

	cv::Mat fg_means_;
	cv::Mat bg_means_;
	cv::Mat fg_weights_;
	cv::Mat bg_weights_;
	std::vector<cv::Mat> fg_covs_;
	std::vector<cv::Mat> bg_covs_;
	std::vector<cv::Mat> fg_inv_covs_;
	std::vector<cv::Mat> bg_inv_covs_;
	std::vector<double> fg_dets_;
	std::vector<double> bg_dets_;

	cv::Mat src_img_;
	cv::Mat src_img_real_;
	cv::Mat fg_seed_img_;
	cv::Mat bg_seed_img_;
	cv::Mat n_links_img_;
	cv::Mat t_links_img_;
	cv::Mat n_links_img_real_;
	cv::Mat t_links_img_real_;
	
	cv::Mat trimap_img_;
	cv::Mat segmentation_img_;
	cv::Mat segmentation_fg_img_;
	cv::Mat segmentation_bg_img_;
	cv::Mat segmentation_fg_mask_;
	cv::Mat segmentation_bg_mask_;

	bool fg_seed_on_;
	bool bg_seed_on_;
	
	cv::Mat fg_prob_;
	cv::Mat bg_prob_;

	std::vector<cv::Mat> n_links_;
	cv::Mat fg_t_links_;
	cv::Mat bg_t_links_;


	// Graph for graph cuts
	GraphType *graph_;
	GraphNodes *nodes_;

	void init_graph();	


	void create_trimap();
	double compute_beta();
	double compute_lambda();
	double distance(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1);
	double distance2(const cv::Vec3d& c1, const cv::Vec3d& c2);
	void compute_N_links();
	cv::Mat invert_3x3_matrix(const cv::Mat& m);
	void compute_dets_and_inv_covs();
	double compute_prob(const cv::Vec3d& color, const cv::Mat& means, const cv::Mat& weights, const std::vector<double>& dets, const std::vector<cv::Mat>& inv_covs);
	void compute_T_links();

	void process_segmentation();

	void create_images();
};