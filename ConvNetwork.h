#pragma once
#include "ConvNetworkLayer.h"
#include <vector>
#include "LearningPair.h"
#include "MatrixOperations.h"
#include "ConvNetworkLayer.h"
#include "ConvolutionalLayer.h"
#include "SubsamplingLayer.h"
#include "MPL_Layer.h"
#include "OutputLayer.h"
class ConvNetwork
{
public:
	ConvNetwork(int inp_w,int inp_h,int f_l_maps_num,int f_l_k_w,int f_l_k_h, int s_l_maps_num,int s_l_k_w,int s_l_k_h,
	int mpl1_neurs_num,int mpl2_neurs_num,int outp_vector_lenght);
	~ConvNetwork(void);
	void learn(std::vector<LearningPair>learning_pairs,int lp_num,float precision,float safe_counter);
	void recognize(float **input);
	void print_info();
	std::vector<ConvNetworkLayer*>layers_ptrs;
	MPL_Layer* mpl_first_ptr;
	MPL_Layer* mpl_sec_ptr;
	OutputLayer *outp_ptr;
	int output_vector_lenght;
	float precision;
private: 
	int conv_layers_num;
	int iterations_number;
	void send_signal_front();
	void send_signal_back(float*expected_output);
	void correct_weights();
};

