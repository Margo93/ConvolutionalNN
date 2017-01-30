#include "stdafx.h"
#include "SubsamplingLayer.h"
#include "MatrixOperations.h"


SubsamplingLayer::SubsamplingLayer(ConvNetworkLayer* prev_conv_layer,int _next_layers_maps_number):ConvNetworkLayer(prev_conv_layer->feature_maps_number)
{
	input_w = prev_conv_layer->output_w;
	input_h = prev_conv_layer->output_h;
	//compression koef = 2
	//"valid" boundary mode
	output_w = input_w/2;
	output_h = input_h/2;
	next_layer_maps_number = _next_layers_maps_number;
	aj = MatrixOperations::create_vector(feature_maps_number);
	//feature maps creation
    feature_maps.reserve(feature_maps_number);
	errors.reserve(feature_maps_number);
	outputs.reserve(feature_maps_number);
			for(int k=0;k<feature_maps_number;k++)
			{  
				push_back_feature_map(prev_conv_layer->outputs[k]);
		    }
}


SubsamplingLayer::~SubsamplingLayer(void)
{
	inputs.clear();
	MatrixOperations::delete_vector(aj);
	delete_all_feature_maps();
}

void SubsamplingLayer::connect_inputs(std::vector<float**> new_inputs)
{
	for(int i=0; i<feature_maps_number;i++)
		inputs[i] = new_inputs[i];
}

void SubsamplingLayer::get_output()
{
	for(int j=0;j<feature_maps_number;j++)
	{
		feature_maps[j].get_output();
	}
}

void SubsamplingLayer::get_errors_from_mpl(std::vector<float **>packed_errors_from_mpl)
{
	for(int j=0;j<feature_maps_number;j++)
	{
		feature_maps[j].get_map_error_from_mpl(packed_errors_from_mpl[j]);
		errors[j]=feature_maps[j].error;
		bj[j]=feature_maps[j].b;
		feature_maps[j].change_a();
		aj[j]=feature_maps[j].a;
	}
}

void SubsamplingLayer::set_link_with_next_conv_layer(int current_layer_map_id,ConvolutionalLayer *next_layer,
		std::vector<int> *connected_maps_from_next_layer)
{
	for(int i=0; i<connected_maps_from_next_layer->size();i++)
	{
		feature_maps[current_layer_map_id].conv_maps_next_layer.push_back(&(next_layer->feature_maps[(*connected_maps_from_next_layer)[i]]));
	}
}

void SubsamplingLayer::get_error()
{ //errors.clear();
	for(int j=0;j<feature_maps_number;j++)
	{
		feature_maps[j].get_map_error_from_convolutional();
	    errors[j]=feature_maps[j].error;
		bj[j]=feature_maps[j].b;
		feature_maps[j].change_a();
		aj[j]=feature_maps[j].a;
	}
}

void SubsamplingLayer::print_output()
{
	for(int j=0;j<feature_maps_number;j++)
	{
		MatrixOperations::print_matrix(outputs[j],output_w,output_h);
	}
}

void SubsamplingLayer::push_back_feature_map(float **inp)
{
	feature_maps.push_back(SubsamplingFeatureMap(input_w,input_h,output_w,output_h,inp,next_layer_maps_number));
	feature_maps.back().output = MatrixOperations::create_2d_matrix(output_w,output_h);
  	feature_maps.back().error = MatrixOperations::create_2d_matrix(output_w,output_h);
  	feature_maps.back().non_activated_stages = MatrixOperations::create_2d_matrix(output_w,output_h);
	errors.push_back(feature_maps.back().error);
	outputs.push_back(feature_maps.back().output);
}

void SubsamplingLayer::delete_all_feature_maps()
{
	for(int k=0;k<feature_maps_number;k++)
		   {
	        MatrixOperations::delete_2d_matrix(feature_maps[k].error,output_w,output_h);
			MatrixOperations::delete_2d_matrix(feature_maps[k].output,output_w,output_h);
			MatrixOperations::delete_2d_matrix(feature_maps[k].non_activated_stages,output_w,output_h);
			}
}