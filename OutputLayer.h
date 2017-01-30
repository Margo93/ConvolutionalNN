#pragma once
#include "backpnnlayer.h"
#include "MPL_Layer.h"
class OutputLayer :
	public BackPNNLayer
{
public:
	OutputLayer(MPL_Layer *prev_mpl,int neur_number);
	OutputLayer(void);
	~OutputLayer(void);
	void get_error(float *expected_output);
};

