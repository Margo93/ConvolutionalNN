#include "stdafx.h"
#include "ConvNetworkLayer.h"
#include "MatrixOperations.h"


ConvNetworkLayer::ConvNetworkLayer(int f_maps_number)
{
	feature_maps_number = f_maps_number;
    outputs.reserve(feature_maps_number);
	errors.reserve(feature_maps_number);
	//initial offset
	bj = MatrixOperations::create_vector(f_maps_number);
	for(int j=0; j<f_maps_number;j++)
		bj[j]=0.01f;
}


ConvNetworkLayer::~ConvNetworkLayer(void)
{
	MatrixOperations::delete_vector(bj);
	outputs.clear();
	errors.clear();
}

void ConvNetworkLayer::get_output(){}
void ConvNetworkLayer::correct_weights(){}
void ConvNetworkLayer::get_error(){}
void ConvNetworkLayer::get_error(ConvNetworkLayer *prev_layer){}
void ConvNetworkLayer::print_output(){}
void ConvNetworkLayer::connect_input(float **new_input){}
void ConvNetworkLayer::change_input(float **new_input){}
void ConvNetworkLayer::get_errors_from_mpl(std::vector<float **>packed_errors_from_mpl){}