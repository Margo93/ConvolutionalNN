#include "stdafx.h"
#include "MPL_Layer.h"
#include "MatrixOperations.h"
#include "ActFuncs.h"
#include <iostream>

MPL_Layer::MPL_Layer(int inps_num,int inp_w, int inp_h,int neurons_num,int next_layer_neurons_num):BackPNNLayer(neurons_num)
{
	is_first = true;
	input_w = inp_w;
	input_h = inp_h;
	//total number of previous layer's neurons
	full_inputs_number = inps_num*inp_w*inp_h;
	//full connection
	input = MatrixOperations::create_vector(full_inputs_number);
	inputs_number = inps_num;
	inputs.reserve(inps_num);
	packed_errors_for_prev_subs.reserve(inps_num);
	for(int i=0;i<inps_num;i++)
		packed_errors_for_prev_subs.push_back(MatrixOperations::create_2d_matrix(inp_w,inp_h));
	next_layer_outputs_number = next_layer_neurons_num;
	this->weights = MatrixOperations::create_2d_matrix(full_inputs_number,outputs_number);
	MatrixOperations::init_matrix_random(weights,full_inputs_number,outputs_number);
}

MPL_Layer::MPL_Layer(MPL_Layer *prev_mpl,int neurons_num,int next_layer_neurons_num):BackPNNLayer(neurons_num)
{
	is_first = false;
	full_inputs_number = prev_mpl->outputs_number;
	//full connection
	input = prev_mpl->output;
	inputs_number = 1;
	next_layer_outputs_number = next_layer_neurons_num;
	this->weights = MatrixOperations::create_2d_matrix(full_inputs_number,outputs_number);
	MatrixOperations::init_matrix_random(weights,full_inputs_number,outputs_number,neurons_num);
}

MPL_Layer::MPL_Layer():BackPNNLayer()
{

}


MPL_Layer::~MPL_Layer(void)
{ 
	MatrixOperations::delete_2d_matrix(weights,full_inputs_number,neurons_number);
	if(is_first)
	{
		MatrixOperations::delete_vector(input);
		for(int i=0;i<inputs_number;i++)
		MatrixOperations::delete_2d_matrix(packed_errors_for_prev_subs[i],input_w,input_h);
	}
}

void MPL_Layer::connect_inputs(std::vector<float**>inps)
{
	for(int i=0;i<inputs_number;i++)
		inputs.push_back(inps[i]);
}

void MPL_Layer::repack_inputs()
{
	int inp_id=0;
	for(int k=0; k<inputs_number;k++)
	{
		for(int j=0; j<input_h;j++)
		{
			for(int i=0; i<input_w;i++)
			{ 
				input[inp_id]=inputs[k][i][j];
				inp_id++;
			}
		}
	
	}
}

void MPL_Layer::get_output_from_maps()
{
	repack_inputs();
	get_output();
}

void MPL_Layer::get_output_from_mpl()
{
	get_output();
}

void MPL_Layer::get_error(float *sigma_next_layer,float**weights_next_layer)
{//err = W_transponed * sigma_next_layer * f_derived(ul)
	float part_summ_error;

	for(int p=0; p<outputs_number;p++)
	{ part_summ_error =0;
		for(int q=0; q<next_layer_outputs_number;q++)
		{
			part_summ_error+=sigma_next_layer[q]*weights_next_layer[p][q];
		}
		error[p] = part_summ_error*ActFuncs::f_act_sigmoidal_deriv(non_activated_stages[p]);
   }
}

void MPL_Layer::pack_error_for_subs()
{
	int id=0;
	for(int psn=0; psn<inputs_number;psn++)
	{
		for(int j=0; j< input_h;j++)
		{
			for(int i=0;i<input_w;i++)
			{	packed_errors_for_prev_subs[psn][i][j]=0;
			for(int u=0;u<outputs_number;u++)
				packed_errors_for_prev_subs[psn][i][j]+=error[u]*weights[id][u];

			id++;
			}
		}
	}
}