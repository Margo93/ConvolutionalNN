#pragma once
#include <vector>
class ConvNetworkLayer
{
public:
	ConvNetworkLayer(int f_maps_number);
	~ConvNetworkLayer(void);
	int feature_maps_number;
	float *bj;
	std::vector<float**>errors;
	std::vector<float**>outputs;
	int output_w;
	int output_h;
	virtual void get_output();
	virtual void correct_weights();
	virtual void get_error();
	virtual void get_error(ConvNetworkLayer *prev_layer);
	virtual void print_output();
	virtual void connect_input(float **new_input);
	virtual void change_input(float **new_input);
	virtual void get_errors_from_mpl(std::vector<float **>packed_errors_from_mpl);
};

