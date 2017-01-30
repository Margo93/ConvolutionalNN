#pragma once
#include "featuremap.h"
class ConvolutionalFeatureMap :
	public FeatureMap
{
public:
	ConvolutionalFeatureMap(int w,int h,int outp_w,int outp_h);
	~ConvolutionalFeatureMap(void);
	void add_input_full_connection(float **input);
	virtual void get_output();
	virtual void correct_weights();
	void get_map_error(float ** sigma_next_layer);
private:
	void convolution(float **_input);
	void fold(float **_input,float **_destination);
	void fold_error(float **_input,float **_destination);
	void upsample(float **sigma_next_l,float**_destination);
};