#pragma once
class BackPNNLayer
{
public:
	BackPNNLayer::BackPNNLayer(void);
	BackPNNLayer(int neur_number);
	~BackPNNLayer(void);
	float **weights;
	float *output;
	float *error;
	int outputs_number;
    int full_inputs_number;
	void get_output();
	void correct_weights();
    void print_input();
	void print_output();
	void print_rounded_output();
    void print_error();
	void print_weights();
	void set_learning_speed(float);
	int neurons_number;
	float *input;
	float *non_activated_stages;
	float learning_speed;
};

