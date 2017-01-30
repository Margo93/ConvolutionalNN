#pragma once
#include <vector>
#include "SubsamplingFeatureMap.h"
#include "ConvNetworkLayer.h"
#include "ConvolutionalLayer.h"

class SubsamplingLayer: public ConvNetworkLayer
{
public:
	SubsamplingLayer(ConvNetworkLayer*prev_conv_layer,int _next_layer_maps_number);
	~SubsamplingLayer(void);
	std::vector<float**>inputs;
    void connect_inputs(std::vector<float**>inputs);
	std::vector<SubsamplingFeatureMap> feature_maps;
	virtual void get_output() override;
	virtual void get_errors_from_mpl(std::vector<float **>packed_errors_from_mpl) override;
	virtual void get_error() override;
	virtual void print_output() override;
	void set_link_with_next_conv_layer(int current_layer_map_id,ConvolutionalLayer *next_layer,
		std::vector<int> *connected_maps_from_next_layer);
	void push_back_feature_map(float **inp);
	void delete_all_feature_maps();
	int input_w;
	int input_h;
	private:
		float *aj;
		int next_layer_maps_number;
};


