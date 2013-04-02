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

#include <math.h>

#include <fstream>

#include <unistd.h>

#include <getopt.h>

#include "pca.h"

using namespace std;

int main(int argc, char *argv[])
{
	FILE *stream;
	int n, m; 

	float in_value;

	n = atoi(argv[2]); //# row
	m = atoi(argv[3]); //# column

	stream = fopen(argv[1], "r");
	
	float *data = (float *)malloc(sizeof(float)*(m*n+3)); //The first two floats stores n and m
	*data = 0; //pass number
	*(data + 1) = n;
	*(data + 2) = m;

	double beforeload = rtclock();
	float *ptr = data + 3;	

	for(int i = 0; i<m*n; i++)
	{
		fscanf(stream, "%f", &in_value);
		*ptr = in_value;
		ptr++;
	}
	double afterload = rtclock();

	cout<<"load time: "<<afterload - beforeload<<endl;

	double beforecor = rtclock();
	corcol(data+3, n, m);
	double aftercor = rtclock();

	cout<<"cor time: "<<aftercor - beforecor<<endl;
	
	unsigned int input_size = (m*m - m)/2;
	pos_t *input = new pos_t[input_size];
	unsigned int count = 0;

	for(int i = 0; i<m; i++)	
	{
		for(int j = i+1; j<m; j++)
		{
			input[count].x = i;
			input[count].y = j;
			count++;
		}
	}

	
	Scheduler scheduler(data, sizeof(float)*(m*n+3), input, input_size, sizeof(pos_t), false); 
	cout<<"The size of data: "<<sizeof(float)*m*n<<endl;
	cout<<"The size of input: "<<sizeof(pos_t)*input_size<<endl;
	scheduler.do_mapreduce();


	unsigned total_size_of_keys = scheduler.get_total_key_size();
	unsigned num_of_keys = scheduler.get_total_key_num();

	
	cout<<"Mapreduce finished.."<<endl;

	cout<<"Total size of keys: "<<total_size_of_keys<<endl;
	cout<<"Num of keys: "<<num_of_keys<<endl;

	scheduler.destroy();
	delete []input;
	fclose(stream);
	free(data);
}
