#pragma once
#include "backpnnlayer.h"
#include <vector>

class MPL_Layer :
	public BackPNNLayer
{
public:
	MPL_Layer(int inps_num,int inp_w, int inp_h,int neurons_num,int next_layer_neurons_num);
	MPL_Layer(MPL_Layer *prev_mpl,int neurons_num,int next_layer_neurons_num);
	MPL_Layer(void);
	~MPL_Layer(void);
	void connect_inputs(std::vector<float**>);
	void repack_inputs();
	void get_output_from_maps();
	void get_output_from_mpl();
	void get_error(float *sigma_next_layer,float**weights_next_layer);
	std::vector<float**> packed_errors_for_prev_subs;
	void pack_error_for_subs();
    private:
	std::vector<float**> inputs;
	int inputs_number;
	int input_w;
	int input_h;
	int next_layer_outputs_number;
	bool is_first;
};

