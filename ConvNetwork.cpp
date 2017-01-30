#include "stdafx.h"
#include "ConvNetwork.h"
#include <iostream>
#include<ctime>


ConvNetwork::ConvNetwork(int inp_w,int inp_h,int f_l_maps_num,int f_l_k_w,int f_l_k_h, int s_l_maps_num,int s_l_k_w,int s_l_k_h,
						 int mpl1_neurs_num,int mpl2_neurs_num,int outp_vector_lenght)
{          
	        conv_layers_num = 4;
			output_vector_lenght = outp_vector_lenght;
			//
			//layers' creation
			layers_ptrs.reserve(conv_layers_num);
			//l1
			static ConvolutionalLayer l1 = ConvolutionalLayer(f_l_maps_num,f_l_k_w,f_l_k_h,inp_w,inp_h);
			static ConvolutionalLayer*l1_ptr =&(l1);
			layers_ptrs.push_back(l1_ptr);

			static SubsamplingLayer l2 = SubsamplingLayer(l1_ptr,4);
			static SubsamplingLayer*l2_ptr =&(l2);
			layers_ptrs.push_back(l2_ptr);

			static ConvolutionalLayer l3 = ConvolutionalLayer(4,2,2,l2.output_w,l2.output_h);
			static ConvolutionalLayer*l3_ptr =&(l3);
			layers_ptrs.push_back(l3_ptr);

			static SubsamplingLayer l4 = SubsamplingLayer(l3_ptr,3);
			static SubsamplingLayer*l4_ptr =&(l4);
			layers_ptrs.push_back(l4_ptr);
			//layers' connection
			//each map of second layer is connected to n/2 maps of third layer
			//set toopology for second layer
			std::vector<int> topology;
			std::vector<int> *topology_ptr=&(topology);
			int counter=0;
			for(int i=0;i<l2.feature_maps_number;i++)
			{
				for(int j=0;j<l3.feature_maps_number/l2.feature_maps_number;j++)
					{
						topology.push_back(counter);
						counter++;
					}
				l2.set_link_with_next_conv_layer(i,l3_ptr,topology_ptr);
				topology.clear();			
			}
			counter=0;

			//set toopology for third layer
			for(int i=0; i<l3.feature_maps_number;i++)
			{
				counter = i*l2.feature_maps_number/l3.feature_maps_number;
				l3.feature_maps[i].add_input_full_connection(l2.feature_maps[counter].output);
			
			}

			//create bpnn network
			static MPL_Layer mpl_first = MPL_Layer(l4.feature_maps_number,l4.output_w,l4.output_h,mpl1_neurs_num,mpl2_neurs_num);
			mpl_first_ptr = &(mpl_first);
			static MPL_Layer mpl_sec =MPL_Layer(mpl_first_ptr,mpl2_neurs_num,outp_vector_lenght);
			mpl_sec_ptr = &(mpl_sec);
			static OutputLayer output =OutputLayer(mpl_sec_ptr,outp_vector_lenght);
			outp_ptr = &(output);
			//link bpnn and conv layers
			mpl_first.connect_inputs(l4.outputs);

	//init precision
    precision = 0.1f;
	iterations_number=0;
}


ConvNetwork::~ConvNetwork(void)
{
	layers_ptrs.clear();
}


void ConvNetwork::learn(std::vector<LearningPair>learning_pairs,int lp_num,float _precision,float safe_counter)
{ int cycle_counter = 0;
  int lp=0;
  srand(time(0));
	
	while (iterations_number<safe_counter)
	{ 
	lp =(int)(rand()%lp_num);

			layers_ptrs[0]->change_input(learning_pairs[lp].input);

			send_signal_front();
			send_signal_back(learning_pairs[lp].expected_output);
			correct_weights();

			precision = abs(learning_pairs[lp].expected_output[0] - mpl_sec_ptr->output[0]);
			for(int i=1; i<output_vector_lenght;i++)
				{
					float cur_prec =  abs(learning_pairs[lp].expected_output[i]-mpl_sec_ptr->output[i]);
					if(cur_prec>precision)
						precision=cur_prec;
				}

			iterations_number++;
			cycle_counter++;
		//	if(cycle_counter==1000)

	}

}

void ConvNetwork::recognize(float** inp)
{
		layers_ptrs[0]->change_input(inp);

	for(int i=0;i<conv_layers_num;i++)
		layers_ptrs[i]->get_output();

	mpl_first_ptr->get_output_from_maps();
	mpl_sec_ptr->get_output_from_mpl();
	outp_ptr->get_output();

	outp_ptr->print_rounded_output();
}

void ConvNetwork::send_signal_front()
{
	for(int i=0;i<conv_layers_num;i++)
		{layers_ptrs[i]->get_output();
	}

	mpl_first_ptr->get_output_from_maps();
	mpl_sec_ptr->get_output_from_mpl();
	outp_ptr->get_output();
}

void ConvNetwork::send_signal_back(float*expected_output)
{
	outp_ptr->get_error(expected_output);
	mpl_sec_ptr->get_error(outp_ptr->error,outp_ptr->weights);
	mpl_first_ptr->get_error(mpl_sec_ptr->error,mpl_sec_ptr->weights);
	mpl_first_ptr->pack_error_for_subs();

	layers_ptrs[3]->get_errors_from_mpl(mpl_first_ptr->packed_errors_for_prev_subs);
	layers_ptrs[2]->get_error(layers_ptrs[3]);
	layers_ptrs[1]->get_error();
	layers_ptrs[0]->get_error(layers_ptrs[1]);
}

void ConvNetwork::correct_weights()
{ 	
	layers_ptrs[0]->correct_weights();
	layers_ptrs[2]->correct_weights();

	mpl_first_ptr->correct_weights();
	mpl_sec_ptr->correct_weights();
	outp_ptr->correct_weights();
}

void ConvNetwork::print_info()
{
	std::cout<<std::endl<<"total iterations number: "<<iterations_number<<std::endl;
	std::cout<<"final precision: "<<precision<<std::endl;
}
