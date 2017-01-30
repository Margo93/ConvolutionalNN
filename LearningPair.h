#pragma once
#include <vector>
class LearningPair
{
public:
	LearningPair(int ipp_w,int inp_h,int outp_l,float** inp,float *expected_outp);
	~LearningPair(void);
	float** input;
	float *expected_output;
private:
	int input_w;
	int input_h;
	int output_lenght;
};

