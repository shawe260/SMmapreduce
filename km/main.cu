#include <iostream>
using namespace std;
#include <fstream>
#include "lib/scheduler.cu"

#ifndef KMEANS
#define KMEANS
#include "kmeans.h"
#endif

#define GRIDSZ 1000

struct kmeans_value
{
	float dim0;
	float dim1;
	float dim2;
	float num;
	float dist;
};

int main()
{
	ifstream input;
	char filename[] = "data";
	//The first K points are cluster centers
	float *points = (float *)malloc(sizeof(float)*DIM*K + sizeof(float)*DIM*BSIZE);

    srand(2006);
	//Generate cluster centers
	for(int i = 0; i<K; i++)
		for(int j = 0; j<DIM; j++)
			points[i*DIM+j]= rand()%GRIDSZ;
	
	//Every point has a offset relative to the start of the points data
	unsigned int *offsets = (unsigned int *)malloc(sizeof(unsigned int)*BSIZE); 

	input.open(filename);
	int exhaust;
	input.read((char *)&exhaust, sizeof(int));
	cout<<exhaust<<endl;
	input.read((char *)&exhaust, sizeof(int));
	cout<<exhaust<<endl;

	cout<<"The first two integers are exhausted..."<<endl<<endl;

	double tool;

	//Points is the global data, we should also generate a logical(offset) data, which is used to reference the global data
	//Load data points
	
	cout<<"The BSIZE is: "<<BSIZE<<endl;
	cout<<"Loading data... "<<endl;
	double beforeload = rtclock();
	for(int i = 0; i< BSIZE*DIM; i++)
	{
		input.read((char *)&tool, sizeof(double));
		points[i+DIM*K] = (float)tool;

		if(i%DIM==0)
		offsets[i/DIM] = i+DIM*K; //The DIM*K is the number of floats used in cluster centers
	}
	input.close();
	double afterload = rtclock();

	cout<<"Load done..."<<endl;
	cout<<"Load time: "<<(afterload-beforeload)<<endl<<endl;
	cout<<"Doing mapreduce..."<<endl;

	Scheduler scheduler(points, sizeof(float)*DIM*(BSIZE+K), offsets, BSIZE, sizeof(unsigned int), false); 

	double beforemap = rtclock();
	scheduler.do_mapreduce();
	double aftermap = rtclock();

	cout<<"The time of mapreduce: "<<aftermap-beforemap<<endl;

	unsigned total_size_of_keys = scheduler.get_total_key_size();
	unsigned num_of_keys = scheduler.get_total_key_num();

	struct output output = scheduler.get_output();

	char *output_keys = output.output_keys;
	char *output_vals = output.output_vals;
	unsigned int *key_index = output.key_index;
	unsigned int *val_index = output.val_index;

	cout<<"*******************************************"<<endl;

	int total_num = 0;
	for(int m = 0; m < num_of_keys; m++)
	{
		char *key_address = output_keys + key_index[m];
		char *val_address = output_vals + val_index[m];
		struct kmeans_value value = *(struct kmeans_value *)val_address;
		int number = (int)value.num;
		float dist = value.dist;
		cout<<*(int *)key_address<<": ";
		cout<<"Average point: ("<<value.dim0/number<<", "<<value.dim1/number<<", "<<value.dim2/number<<")";
		printf("\t Number of points: %d", number);
		printf("\t Dist: %f", dist);
		total_num+=number;
		cout<<endl;
	}

	cout<<"*******************************************"<<endl;
	printf("The total number of points: %d\n", total_num);

	scheduler.destroy();

	free(points);
	free(offsets);
}
