#pragma once
#include <vector>
#include "LearningPair.h"
#include "MPL_Layer.h"
#include "OutputLayer.h"
#include "MPL_Layer.h"
#include "OutputLayer.h"
class BPNN
{
public:
	BPNN(int input_maps_num,int inp_map_w,int inp_map_h,int first_mpl_neur_num,int sec_mpl_neur_num,int outp_length);
	~BPNN(void);
	void recognize(std::vector<float**>);
	void make_iteration(LearningPair* cur_learning_pair);
	void print_info();
	MPL_Layer* mpl_first_ptr;
	MPL_Layer* mpl_sec_ptr;
	OutputLayer *outp_ptr;
	int result_vector_lenght;
	void send_signal_front();
	void send_signal_back(float *expected_outp);
	void correct_weights();
};

