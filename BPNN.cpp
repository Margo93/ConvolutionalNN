#include "stdafx.h"
#include "BPNN.h"
#include <math.h>
#include <iostream>



BPNN::BPNN(int input_maps_num,int inp_map_w,int inp_map_h,int first_mpl_neur_num,int sec_mpl_neur_num,int outp_length)
{
	static MPL_Layer mpl_first = MPL_Layer(input_maps_num,inp_map_w,inp_map_h,first_mpl_neur_num,sec_mpl_neur_num);
	mpl_first_ptr = &(mpl_first);
	static MPL_Layer mpl_sec =MPL_Layer(mpl_first_ptr,sec_mpl_neur_num,outp_length);
    mpl_sec_ptr = &(mpl_sec);
	static OutputLayer output =OutputLayer(mpl_sec_ptr,outp_length);
	outp_ptr = &(output);
	result_vector_lenght =outp_length;
}


BPNN::~BPNN(void)
{
	
}


void BPNN::recognize(std::vector<float**> inp)
{
	mpl_first_ptr->connect_inputs(inp);
	send_signal_front();
	outp_ptr->print_rounded_output();
}

void BPNN::send_signal_front()
{
	mpl_first_ptr->get_output_from_maps();
	mpl_sec_ptr->get_output_from_mpl();
	outp_ptr->get_output();
	
}

void BPNN::send_signal_back(float *expected_outp)
{
	outp_ptr->get_error(expected_outp);
	mpl_sec_ptr->get_error(outp_ptr->error,outp_ptr->weights);
	mpl_first_ptr->get_error(mpl_sec_ptr->error,mpl_sec_ptr->weights);
	mpl_first_ptr->pack_error_for_subs();

}

void BPNN::correct_weights()
{
	mpl_first_ptr->correct_weights();
	mpl_sec_ptr->correct_weights();
	outp_ptr->correct_weights();
}

