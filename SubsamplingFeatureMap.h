#pragma once
#include "featuremap.h"
#include <vector>

class SubsamplingFeatureMap :
	public FeatureMap
{
public:
	SubsamplingFeatureMap(int width,int height,int outp_w,int outp_h,float**inp,int connected_next_l_maps_number);
	~SubsamplingFeatureMap(void);
	virtual void get_output();
	void get_map_error_from_mpl(float **sigma_next_layer);
	void get_map_error_from_convolutional();
	float a;
	void change_a();
	int next_layer_maps_number;
	void attach_next_conv_layer_maps(std::vector<FeatureMap*>conv_maps_next_l);
	std::vector<FeatureMap*>conv_maps_next_layer;
private:
	float **deriv_non_activated_stages;
	float **input;
	//?
	void subsample(float**_source,float**_destination);
	void fold(float **_input,float**_weights,float **_destination,int _k_w,int _k_h,int n_l_inp_w,int n_l_inp_h);

};
