#include "stdafx.h"
#include "LearningPair.h"


LearningPair::LearningPair(int inp_w,int inp_h,int outp_l,float** inp,float *expected_outp)
{
	input_w = input_w;
	input_h = inp_h;
	output_lenght = outp_l;
	input = inp;
	expected_output = expected_outp;
}


LearningPair::~LearningPair(void)
{
}
