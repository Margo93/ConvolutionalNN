#pragma once
#include <vector>

class FeatureMap
{
public:
	FeatureMap(int width,int height,int outp_w,int outp_h);
	~FeatureMap(void);
	float **output;
	float **error;
	float **weights;
	std::vector<float **> inputs;
	float b;
	virtual void get_output();
    virtual void correct_weights();
	int w;
	int h;
	//output sizes depend on boundary processing methods(valid/same/full) and its' boundary effects
	int output_w;
	int output_h;
	float **non_activated_stages;

};
