#include <iostream>
using namespace std;
#include <fstream>
#include "lib/scheduler.cu"

#ifndef KMEANS
#define KMEANS
#include "kmeans.h"
#endif

#include "lib/common.h"

int main()
{
	ifstream input;
	char filename[] = "data";


	//The first point is the test sample
	float *points = (float *)malloc(sizeof(float)*(DIM+1) + sizeof(float)*(DIM+1)*BSIZE);

	//initialize the test sample
	points[0] = -1;
	points[1] = 100;
	points[2] = 100;
	points[3] = 100;
	
	//Every point has an offset relative to the start of the points data
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

	float *data = points+DIM+1; //data is the start of the trained points
	for(int i = 0; i < BSIZE; i++)
	{
		offsets[i] = DIM+1+i*(DIM+1);
		data[0] = i;
		for(int j = 1; j<DIM+1; j++)
		{
			input.read((char *)&tool, sizeof(double));
			data[j] = (float)tool;
		}
		data=data+DIM+1;
	}
	
	input.close();
	double afterload = rtclock();

	cout<<"Load done..."<<endl;
	cout<<"Load time: "<<(afterload-beforeload)<<endl<<endl;
	cout<<"Doing mapreduce..."<<endl;

	Scheduler scheduler(points, sizeof(float)*(DIM+1)*(BSIZE+1), offsets, BSIZE, sizeof(unsigned int), sizeof(float)); 

	scheduler.do_mapreduce();


	double beforesort = rtclock();
	scheduler.sort_object();
	double aftersort = rtclock();

	cout<<"Sort time: "<<aftersort-beforesort<<endl;

	scheduler.trunk_object(SORT_REMAIN);

	unsigned total_size_of_keys;
	unsigned num_of_keys;

	scheduler.get_sizes(&total_size_of_keys, &num_of_keys);

	cout<<"Total size of keys: "<<total_size_of_keys<<endl;
	cout<<"Num of keys: "<<num_of_keys<<endl;

	void *key_array_h = (void *)malloc(total_size_of_keys);
	int *key_offset_array_h = (int *)malloc(num_of_keys*sizeof(int));
	void *value_array_h = (void *)malloc(sizeof(float)*num_of_keys);

	scheduler.copy_data(key_array_h, key_offset_array_h, value_array_h, total_size_of_keys, num_of_keys, sizeof(float)*num_of_keys);

	float *point_start = points+DIM+1;
	for(int i = 0; i<num_of_keys; i++)
	{
		char *key_address = (char *)key_array_h+key_offset_array_h[i];
		int key = (int)*(float *)key_address;
		cout<<key<<": "<<"\t\t";
		char *point_addr = (char *)value_array_h+sizeof(float)*i;
		float distance = ((float *)point_addr)[0];
		float dim0 = (point_start+(DIM+1)*key)[1];
		float dim1 = (point_start+(DIM+1)*key)[2];
		float dim2 = (point_start+(DIM+1)*key)[3];
	//	float dim1 = ((float *)point_addr)[2];
	//	float dim2 = ((float *)point_addr)[3];

		cout<<"Point: ("<<dim0<<", "<<dim1<<", "<<dim2<<")";
		printf("\t\t Distance: %f", distance);
		cout<<endl;
	}

	//printf("The total number of points: %d\n", total_num);

	scheduler.destroy();

	free(key_array_h);
	free(key_offset_array_h);
	free(value_array_h);
	free(points);
	free(offsets);
}
