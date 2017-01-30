#pragma once
#include <vector>
#include "ConvolutionalFeatureMap.h"
#include "ConvNetworkLayer.h"

class ConvolutionalLayer: public ConvNetworkLayer
{
public:
	ConvolutionalLayer(int f_maps_number, int _k_w,int _k_h,int _inp_w,int _inp_h);
	~ConvolutionalLayer(void);
	std::vector<ConvolutionalFeatureMap> feature_maps;
	virtual void connect_input(float **input) override;
	virtual void change_input(float **input) override;
	virtual void get_output() override;
	virtual void get_error(ConvNetworkLayer* next_layer) override;
	virtual void correct_weights() override;
	virtual void print_output() override;
	void push_back_feature_map();
	void delete_all_feature_maps();
	int map_w;
	int map_h;
	int k_w;
	int k_h;
};