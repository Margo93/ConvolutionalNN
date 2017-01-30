#include "stdafx.h"
#include "ConvolutionalFeatureMap.h"
#include "MatrixOperations.h"
#include "ActFuncs.h"
#include <iostream> 


ConvolutionalFeatureMap::ConvolutionalFeatureMap(int w,int h,int outp_w,int outp_h):FeatureMap(w,h,outp_w,outp_h)
{

}


ConvolutionalFeatureMap::~ConvolutionalFeatureMap(void)
{

}

void ConvolutionalFeatureMap::add_input_full_connection(float **input)
{
		inputs.push_back(input);
}

void ConvolutionalFeatureMap::get_output()
{
	for(int i=0;i<inputs.size();i++)
	{
		convolution(inputs[i]);
	}
}

void ConvolutionalFeatureMap::convolution(float **_input)
{
	float **f = MatrixOperations::create_2d_matrix(output_w,output_h);
	fold(_input,f);
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			non_activated_stages[i][j]=f[i][j]+b;
			output[i][j]=ActFuncs::f_act_sigmoidal(f[i][j]+b);
		}
	}

	MatrixOperations::delete_2d_matrix(f,output_w,output_h);
}

void ConvolutionalFeatureMap::fold(float **_input,float **_destination)
{
	//clear destination;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			_destination[i][j]=0;
		}
	}

	float fold_cell=0;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{//summ k,l
			for(int l=0; l<h;l++)
				{
					for(int k=0; k<w;k++)
						fold_cell+=_input[i+k][j+l]*weights[k][l];
				}
			_destination[i][j]+=fold_cell;
			fold_cell=0;
		}
	}

}

void ConvolutionalFeatureMap::get_map_error(float ** sigma_next_layer)
{//err= W^T * sigma_prev*f_deriv(ul)
	float **upsampled_next_l_err = MatrixOperations::create_2d_matrix(output_w,output_h);
	upsample(sigma_next_layer,upsampled_next_l_err);
	for(int j=0; j<output_h;j++)
	{
		for(int i=0;i<output_w;i++)
			error[i][j] = upsampled_next_l_err[i][j]* ActFuncs::f_act_sigmoidal_deriv(non_activated_stages[i][j]);	
	
	}
}

void ConvolutionalFeatureMap::upsample(float **sigma_next_l,float**_destination)
{
	for(int j=0;j<output_h-(output_h%2);j=j+2)
	{
		for(int i=0;i<output_w-(output_w%2);i=i+2)
		{
			_destination[i][j]=sigma_next_l[i/2][j/2];
			_destination[i+1][j]=sigma_next_l[i/2][j/2];
			_destination[i][j+1]=sigma_next_l[i/2][j/2];
			_destination[i+1][j+1]=sigma_next_l[i/2][j/2];
		}
	}

}

void ConvolutionalFeatureMap::correct_weights()
{float ** fold_err = MatrixOperations::create_2d_matrix(h,w);
	
    for(int inp=0; inp<inputs.size();inp++)
	{
		fold_error(inputs[inp],fold_err);

		for(int j=0;j<h;j++)
		{
			for(int i=0; i<w;i++)
				{   
					weights[i][j]+=fold_err[j][i];
					b+=error[i][j];
				}
		}
	 
	}

	MatrixOperations::delete_2d_matrix(fold_err,h,w);
}

void ConvolutionalFeatureMap::fold_error(float **_input,float **_destination)
{
	//clear destination;
	for(int j=0; j<h;j++)
	{
		for(int i=0; i<w;i++)
		{
			_destination[i][j]=0;
		}
	}

	float fold_cell=0;

	for(int j=0;j<w;j++)
	{
		for(int i=0;i<h;i++)
		//summ k,l
		{
			for(int l=0; l<output_h;l++)
			{
				for(int k=0; k<output_w;k++)
					fold_cell+=_input[k+i][l+j] * error[l][k];
			}
		_destination[i][j]+=fold_cell;
		fold_cell=0;
		}
	}
}


