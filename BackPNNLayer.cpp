#include "stdafx.h"
#include "BackPNNLayer.h"
#include "MatrixOperations.h"
#include "ActFuncs.h"
#include<iostream>

BackPNNLayer::BackPNNLayer(int neur_number)
{
	neurons_number = neur_number;
	outputs_number = neur_number;
	full_inputs_number=0;
	error = MatrixOperations::create_vector(neur_number);
	non_activated_stages = MatrixOperations::create_vector(neur_number);
	output=MatrixOperations::create_vector(neur_number);
	learning_speed=0.7f;
}

BackPNNLayer::BackPNNLayer(void)
{
	neurons_number = 0;
	outputs_number = 0;
	full_inputs_number=0;
	error = NULL;
	non_activated_stages = NULL;
	output=NULL;
	learning_speed=0.7f;
}



BackPNNLayer::~BackPNNLayer(void)
{
	//input=NULL;
	MatrixOperations::delete_vector(error);
	MatrixOperations::delete_vector(non_activated_stages);
	MatrixOperations::delete_vector(output);
}

void BackPNNLayer::get_output()
{
	for(int j=0; j<outputs_number;j++)
	{
		//clear output
		non_activated_stages[j]=0;
	    //set new value
		for(int i=0;i<full_inputs_number;i++)
		   {
			   non_activated_stages[j]+=input[i]*weights[i][j];
		   }

		output[j] = ActFuncs::f_act_sigmoidal(non_activated_stages[j]);

	}
}

void BackPNNLayer::correct_weights()
{
	for(int j=0; j<outputs_number;j++)
	{
		for(int i=0; i<full_inputs_number;i++)
			weights[i][j]+=error[j]*input[i]*learning_speed;
	}

}

void BackPNNLayer::print_input()
{MatrixOperations::print_vector(input,full_inputs_number);}

void BackPNNLayer::print_output()
{MatrixOperations::print_vector(output,outputs_number);}

void BackPNNLayer::print_rounded_output()
{MatrixOperations::print_rounded_vector(output,outputs_number);}

void BackPNNLayer::print_error()
{MatrixOperations::print_vector(error,neurons_number);}

void BackPNNLayer::print_weights()
{MatrixOperations::print_matrix(weights,full_inputs_number,outputs_number);}

void BackPNNLayer::set_learning_speed(float new_l_speed)
{learning_speed=new_l_speed;}
