#include "stdafx.h"
#include "MatrixOperations.h"
#include<iostream>
#include<ctime>
#include<iomanip>
#include<cmath>

MatrixOperations::MatrixOperations(void)
{

}


MatrixOperations::~MatrixOperations(void)
{
}

float** MatrixOperations::create_2d_matrix(int w,int h)
{ float **matr=new float*[w];

	for(int i=0;i<w;i++)
		{matr[i]=new float[h];}

  return matr;
}

void MatrixOperations::delete_2d_matrix(float** matrix,int w,int h)
{ 
	for(int i=0;i<w;i++)
		{delete[] matrix[i];}
	delete[]matrix;
}

float* MatrixOperations::create_vector(int len)
{ float *vect=new float[len];
  return vect;
}

void MatrixOperations::delete_vector(float* vect)
{ 
	delete[] vect;
}

void MatrixOperations::set_matrix(float **matrix,float **new_matrix_vals,int w,int h)
{
	for(int j=0;j<h;j++)
	{
	for(int i=0;i<w;i++)
		{
			matrix[i][j]=new_matrix_vals[i][j];
		}
	}
}

void  MatrixOperations::init_matrix_random(float **matrix,int w,int h)
{   float val=0;
    srand(time(0));
	for(int j=0;j<h;j++)
	{
	for(int i=0;i<w;i++)
	{
		val=(float)(rand()%100)/100;
		if((int)(rand()%2)==1)
			val*=(-1);

			matrix[i][j]= val;
		}
	}
}

 //weights in [-1/(2k) ; 1/(2k)] range
void  MatrixOperations::init_matrix_random(float **matrix,int w,int h,int inp_neur_num)
{ 
	float val=0;
    srand(time(0));
	for(int j=0;j<h;j++)
	{
	for(int i=0;i<w;i++)
	{
		//val in [0;1] range
		val=(float)(rand()%100)/100;
		//val in [-1/(2k) ; 1/(2k)] range
	//	val = val/inp_neur_num - 1/(2*inp_neur_num);
		if((int)(rand()%2)==1)
		val*=(-1);
		matrix[i][j]= val;
		}
	}
}

void MatrixOperations::jogging(float **matrix,int w,int h)
{
	 float val=0;
    srand(time(0));
	for(int j=0;j<h;j++)
	{
	for(int i=0;i<w;i++)
	{
		val=(float)(rand()%100)/100;
		if((int)(rand()%2)==1)
			val*=(-1);

			matrix[i][j]+= val;
		}
	}
}


	void MatrixOperations::print_matrix(float **matrix,int w,int h)
	{	std::cout<<std::endl;
		for(int j=0;j<h;j++)
	    {
	        for(int i=0;i<w;i++)
				std::cout<<std::setw(6)<<std::setprecision(4)<<std::fixed<<matrix[i][j]<<"  ";
	      std::cout<<std::endl;
	    }
	    std::cout<<std::endl;
	}

	void MatrixOperations::print_vector(float *vector,int len)
	{
	for(int i=0;i<len;i++)
		{
			std::cout<<std::setw(6)<<std::setprecision(4)<<std::fixed<<vector[i]<<"  ";
		}
	std::cout<<std::endl;
	}
	
		void MatrixOperations::print_rounded_vector(float *vector,int len)
	{
	for(int i=0;i<len;i++)
		{
			std::cout<<std::setw(2)<<std::floor(vector[i]+0.5);
		}
	std::cout<<std::endl;
	}
