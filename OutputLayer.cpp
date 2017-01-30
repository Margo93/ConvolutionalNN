#include "stdafx.h"
#include "OutputLayer.h"
#include "MatrixOperations.h"
#include "ActFuncs.h"


OutputLayer::OutputLayer(MPL_Layer *prev_mpl,int neur_number):BackPNNLayer(neur_number)
{
	input = prev_mpl->output;
	full_inputs_number = prev_mpl->outputs_number;
	weights = MatrixOperations::create_2d_matrix(full_inputs_number,neur_number);
	MatrixOperations::init_matrix_random(weights,full_inputs_number,outputs_number,neur_number);
}

OutputLayer::OutputLayer(void):BackPNNLayer()
{}

OutputLayer::~OutputLayer(void)
{
	MatrixOperations::delete_2d_matrix(weights,full_inputs_number,neurons_number);
}

void OutputLayer::get_error(float *expected_output)
{
	for(int i=0; i<outputs_number;i++)
		error[i]=(expected_output[i]-output[i]) * ActFuncs::f_act_sigmoidal(non_activated_stages[i]);
}