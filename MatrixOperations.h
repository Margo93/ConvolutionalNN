#pragma once
class MatrixOperations
{
public:
	MatrixOperations(void);
	~MatrixOperations(void);
    static float** create_2d_matrix(int w,int h);
	static void delete_2d_matrix(float **matrix,int w,int h);
	static float* create_vector(int len);
	static void delete_vector(float *vect);
	static void set_matrix(float **matrix,float **new_matrix_vals,int w,int h);
	static void init_matrix_random(float **matrix,int w,int h);
	static void init_matrix_random(float **matrix,int w,int h,int inp_neur_num);
	static void print_matrix(float **matrix,int w,int h);
	static void print_vector(float *vector,int len);
	static void print_rounded_vector(float *vector,int len);
	static void jogging(float **matrix,int w,int h);
};

