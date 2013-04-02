#include <iostream>
#include "lib/scheduler.cu"

#ifndef MM 
#define MM 
#include "mm.h"
#endif

#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#include <unistd.h>

#include <getopt.h>

using namespace std;

//--------------------------------------------------
//generate data
//--------------------------------------------------
static float *GenMatrix(int M_ROW_COUNT, int M_COL_COUNT)
{
	float *matrix = (float*)malloc(sizeof(float)*M_ROW_COUNT*M_COL_COUNT);

	srand(10);
	for (int i = 0; i < M_ROW_COUNT; i++)
		for (int j = 0; j < M_COL_COUNT; j++)
			matrix[i*M_COL_COUNT+j] = (float)(rand() % 100);

	return matrix;
}

void TranMatrix(float * m, int row){
	for(int i=0;i<row;i++){
		for(int j=i+1;j<row;j++){
			float tmp=m[i*row+j];
			m[i*row+j]=m[j*row+i];
			m[j*row+i]=tmp;
		}
	}
}

int main(int argc, char *argv[])
{
	cout<<"NUM_BLOCKS: "<<NUM_BLOCKS<<endl;
	cout<<"NUM_THREADS: "<<NUM_THREADS<<endl;
	int dim = DIM;

	pos_t * input = new pos_t[dim*dim];	
	for(int i = 0; i<dim; i++)
	{
		for(int j = 0; j<dim; j++)
		{
			input[i*dim+j].x = i;
			input[i*dim+j].y = j;
		}
	}

	float * matrixA = GenMatrix(dim, dim);
	float * matrixB = GenMatrix(dim, dim);


	/*sequential version*/

//	double beforecpu = rtclock();
//	float *result = new float[dim*dim]; 
//	for(int m = 0; m < dim; m++)
//	for(int n = 0; n < dim; n++)
//	for(int k = 0; k < dim; k++)
//	result[m*dim+n] += matrixA[m*dim+k]*matrixB[k*dim+n];
//	double aftercpu = rtclock();
//	cout<<"cpu time: "<<aftercpu - beforecpu<<endl;
//
//	for(int t = 0; t<10; t++ )
//	cout<<result[t]<<endl;

	float * matdata = (float *)malloc(sizeof(float)*dim*dim*2); 
	memcpy(matdata, matrixA, sizeof(float)*dim*dim);
	float *ptr = matdata+dim*dim;
	memcpy(ptr, matrixB, sizeof(float)*dim*dim);

	Scheduler scheduler(matdata, sizeof(float)*dim*dim*2, input, dim*dim, sizeof(pos_t), false); 
	cout<<"The size of matdata: "<<sizeof(float)*dim*dim*2<<endl;
	cout<<"The size of input: "<<sizeof(pos_t)*dim*dim<<endl;

	double beforemap = rtclock();
	scheduler.do_mapreduce();
	double aftermap = rtclock();

	cout<<"The time of mapreduce: "<<aftermap-beforemap<<endl;

	unsigned total_size_of_keys = scheduler.get_total_key_size();
	unsigned num_of_keys = scheduler.get_total_key_num();

	cout<<"Mapreduce finished.."<<endl;

	cout<<"Total size of keys: "<<total_size_of_keys<<endl;
	cout<<"Total number of keys: "<<num_of_keys<<endl;
	cout<<endl;

	struct output output = scheduler.get_output();

	char *output_keys = output.output_keys;
	char *output_vals = output.output_vals;
	unsigned int *key_index = output.key_index;
	unsigned int *val_index = output.val_index;

	cout<<"*******************************************"<<endl;

	for(int m = 0; m<10; m++)	
	{
		char *key_address = output_keys + key_index[m];	
		char *val_address = output_vals + val_index[m];
		cout<<*(unsigned int *)key_address<<": "<<*(float *)val_address<<endl;
	}

	scheduler.destroy();
	delete []input;
	free(matrixA);
	free(matrixB);
	free(matdata);
}
