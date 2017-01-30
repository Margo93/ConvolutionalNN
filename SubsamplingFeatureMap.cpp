#include "stdafx.h"
#include "SubsamplingFeatureMap.h"
#include "MatrixOperations.h"
#include "ActFuncs.h"
#include <cmath>


SubsamplingFeatureMap::SubsamplingFeatureMap(int width,int height,int outp_w,int outp_h,float**inp,int connected_next_l_maps_number):FeatureMap(width,height,outp_w,outp_h)
{
	input = inp;
	a=1;
	next_layer_maps_number = connected_next_l_maps_number;
}


SubsamplingFeatureMap::~SubsamplingFeatureMap(void)
{

}

void SubsamplingFeatureMap::attach_next_conv_layer_maps(std::vector<FeatureMap*>conv_maps_next_l)
{
	conv_maps_next_layer=conv_maps_next_l;
}

void SubsamplingFeatureMap::get_output()
{
	float **temp = MatrixOperations::create_2d_matrix(output_w,output_h);
	subsample(input,temp);

	for(int j=0; j<output_h;j++)
	{
		for (int i = 0; i < output_w; i++)
		{
			non_activated_stages[i][j]=a*temp[i][j]+b;
			output[i][j]= ActFuncs::f_act_linear(non_activated_stages[i][j],1);
		}
	}

	MatrixOperations::delete_2d_matrix(temp,output_w,output_h);
}

//used: map-pooling and 1/2 compression
void SubsamplingFeatureMap::subsample(float**_source,float**_destination)
{
	for (int j= 0;j < output_h;j++)
	{
		for(int i=0;i<output_w;i++)
		{
			_destination[i][j]=std::max(std::max(std::max(_source[i*2][j*2],_source[i*2+1][j*2]),_source[i*2][j*2+1]),_source[i*2+1][j*2+1]);
		}

	}

}

void SubsamplingFeatureMap::get_map_error_from_mpl(float **sigma_next_layer)
{
	//clear error;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			error[i][j]=0;
		}
	}

	//get new values for error and koefficient b
	b=0;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			error[i][j]+=sigma_next_layer[i][j]*ActFuncs::f_act_linear_deriv(1);
			b+=error[i][j];
		}
	}


}

void SubsamplingFeatureMap::change_a()
{
	float **subs_inp = MatrixOperations::create_2d_matrix(output_w,output_h);
	subsample(input,subs_inp);

	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			a+=error[i][j]*subs_inp[i][j];
		}
	}

	MatrixOperations::delete_2d_matrix(subs_inp,output_w,output_h);
}

void SubsamplingFeatureMap::get_map_error_from_convolutional()
{


	//clear error;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			error[i][j]=0;
		}
	}

	if(next_layer_maps_number>0)
	{//get next layer's parameters
		int next_l_map_w = conv_maps_next_layer[0]->output_w;
		int next_l_map_h = conv_maps_next_layer[0]->output_h;
		int next_l_k_w = conv_maps_next_layer[0]->w;
		int next_l_k_h = conv_maps_next_layer[0]->h;
	
	//create transponed kernels
	std::vector<float**> transponed_kernels;
	for(int m=0; m<next_layer_maps_number;m++)
	{  
		transponed_kernels.push_back(MatrixOperations::create_2d_matrix(next_l_k_h,next_l_k_w));
		for(int j=0; j<next_l_k_h;j++)
		{
			for(int i=0; i<next_l_k_w;i++)
			{
				transponed_kernels[m][j][i]=conv_maps_next_layer[m]->weights[i][j];
			}
		}

	}

	float **summ_fold=MatrixOperations::create_2d_matrix(output_w,output_h);
	float **part_fold=MatrixOperations::create_2d_matrix(output_w,output_h);
	//clear summfold;
	for(int j=0;j<output_h; j++)
	{
		for(int i=0;i<output_w; i++)
		{summ_fold[i][j]=0;
		 part_fold[i][j]=0;
		}
	}


	for(int k=0; k<next_layer_maps_number;k++)
	{
		fold(conv_maps_next_layer[k]->error,transponed_kernels[k],part_fold,next_l_k_w,next_l_k_h,next_l_map_w,next_l_map_h);
		for(int j=0;j<output_h;j++)
		{
			for (int i= 0;i<output_w ; i++)
			{
				summ_fold[i][j]+=part_fold[i][j];
			}
		}
	}

	for(int j=0;j<output_h;j++)
		{
			for (int i= 0;i<output_w ; i++)
			{
				error[i][j]=ActFuncs::f_act_linear_deriv(non_activated_stages[i][j])*summ_fold[i][j];
				b+=error[i][j];
			}
		}
	
	//delete transponed kernels
	for(int m=0; m<next_layer_maps_number;m++)
	{  
		MatrixOperations::delete_2d_matrix(transponed_kernels[m],next_l_k_h,next_l_k_w);
	}
			transponed_kernels.clear();

	MatrixOperations::delete_2d_matrix(summ_fold,output_w,output_h);
	MatrixOperations::delete_2d_matrix(part_fold,output_w,output_h);
}
}
void  SubsamplingFeatureMap::fold(float **_input,float**_weights,float **_destination,
								  int _k_w,int _k_h,int n_l_inp_w,int n_l_inp_h)
{
	//clear destination;
	for(int j=0; j<output_h;j++)
	{
		for(int i=0; i<output_w;i++)
		{
			_destination[i][j]=0;
		}
	}
	//get rid of boundary "cutting" effects of fold

	float **full_input = MatrixOperations::create_2d_matrix(output_w+_k_h-1,output_h+_k_w-1);
	for(int j=0; j<output_h+_k_w-1;j++)
	{
		for(int i=0; i<output_w+_k_h-1;i++)
		 full_input[i][j]=0;

	}



	for(int j=0;j<n_l_inp_h;j++)
	{//centering
		for(int i=0; i<n_l_inp_w;i++)
		{
			full_input[i+(_k_h/2)][j+(_k_w/2)] = _input[i][j];
		}
	}

	float fold_cell=0;

	for(int j=0;j<n_l_inp_h;j++)
	{
		for(int i=0; i<n_l_inp_w;i++)
		{//summ k,l
			for(int l=0;l<_k_w;l++)
				{
					for(int k=0; k<_k_h;k++)
						{
							fold_cell+=full_input[k+i][l+j]*_weights[k][l];
						}
				}
			_destination[i][j]+=fold_cell;
			fold_cell=0;
		}
	}

	MatrixOperations::delete_2d_matrix(full_input,output_w+_k_h-1,output_h+_k_w-1);
}

