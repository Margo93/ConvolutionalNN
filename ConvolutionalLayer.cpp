#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "MatrixOperations.h"

ConvolutionalLayer::ConvolutionalLayer(int f_maps_number, int _k_w,int _k_h,int _inp_w,int _inp_h):ConvNetworkLayer(f_maps_number)
{
	k_w=_k_w;
	k_h=_k_h;
	//"full-mode" boundary effect
	if(_inp_w>_k_w)
		map_w=_inp_w-_k_w+1;
	else 
		map_w =_inp_w;

		if(_inp_h>_k_h)
		map_h=_inp_h-_k_h+1;
	else 
		map_h =_inp_h;

		output_w = map_w;
		output_h=map_h;

		//feature maps creation
		feature_maps.reserve(feature_maps_number);
		errors.reserve(feature_maps_number);
		outputs.reserve(feature_maps_number);
			for(int k=0;k<feature_maps_number;k++)
			{  
				push_back_feature_map();
		    }
}


ConvolutionalLayer::~ConvolutionalLayer(void)
{
	delete_all_feature_maps();
}

//full_connection
void ConvolutionalLayer::connect_input(float **_input)
	
{
			for(int k=0;k<feature_maps_number;k++)
		{
			feature_maps[k].add_input_full_connection(_input);
		}

}

void ConvolutionalLayer::change_input(float **_input)
	
{
			for(int k=0;k<feature_maps_number;k++)
			{
			feature_maps[k].inputs.clear();
			feature_maps[k].add_input_full_connection(_input);
		    }
}

void ConvolutionalLayer::get_output()
{
	for(int j=0;j<feature_maps_number;j++)
	{
		feature_maps[j].get_output();
	}
}

void ConvolutionalLayer::get_error(ConvNetworkLayer* next_layer)
{
		for(int j=0;j<feature_maps_number;j++)
	{
		feature_maps[j].get_map_error(next_layer->errors[j]);
		errors[j]=feature_maps[j].error;
	}
}

void ConvolutionalLayer::correct_weights()
{
	for(int j=0;j<feature_maps_number;j++)
		feature_maps[j].correct_weights();
}

void ConvolutionalLayer::print_output()
{
	for(int j=0;j<feature_maps_number;j++)
		MatrixOperations::print_matrix(outputs[j],map_w,map_h);
}

void  ConvolutionalLayer::push_back_feature_map()
{
			    feature_maps.push_back(ConvolutionalFeatureMap(k_w,k_h,map_w,map_h));
				feature_maps.back().weights = MatrixOperations::create_2d_matrix(k_w,k_h);
				MatrixOperations::init_matrix_random(feature_maps.back().weights,k_w,k_h);
				feature_maps.back().output = MatrixOperations::create_2d_matrix(map_w,map_h);
  			    feature_maps.back().error = MatrixOperations::create_2d_matrix(map_w,map_h);
  				feature_maps.back().non_activated_stages = MatrixOperations::create_2d_matrix(map_w,map_h);
				errors.push_back(feature_maps.back().error);
				outputs.push_back(feature_maps.back().output);
}

void ConvolutionalLayer::delete_all_feature_maps()
{
	for(int k=0;k<feature_maps_number;k++)
		   {MatrixOperations::delete_2d_matrix(feature_maps[k].weights,k_w,k_h);
			MatrixOperations::delete_2d_matrix(feature_maps[k].error,map_w,map_h);
			MatrixOperations::delete_2d_matrix(feature_maps[k].output,map_w,map_h);
			MatrixOperations::delete_2d_matrix(feature_maps[k].non_activated_stages,map_w,map_h);
			}
}