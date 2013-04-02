#include <iostream>
using namespace std;
#include <fstream>
#include <vector>
#include "lib/scheduler.cu"

#ifndef DS_h 
#define DS_h
#include "DS.h"
#endif

int main(int argc, char *argv[])
{
	char *filename = argv[1];
	void *rawbuf;
	unsigned int size;
	map_file(filename, rawbuf, size);
	if(!rawbuf)
	{
		cout<<"error opening file "<<filename<<endl;
		return 1;
	}

	double beforeload = rtclock();

	char *filebuf = new char[size+1]; //The first byte contains the pass number
	memcpy(filebuf+1, rawbuf, size);
	*filebuf = 0;

	vector<unsigned int> offsets;
	offsets.push_back(1);

	for(int i = 1; i<size+1; i++)
	{
		if(filebuf[i]!='\n')
			continue;
		filebuf[i] = '\0';
		//cout<<"i is: "<<i<<endl;
		if(i+1<size+1)
			offsets.push_back(i+1);
	}

	double afterload = rtclock();
	cout<<"Load time: "<<afterload - beforeload<<endl;

	cout<<"The number of offsets: "<<offsets.size()<<endl;

	cout<<"####Doing mapreduce, first pass####"<<endl;

	Scheduler scheduler(filebuf, size+1, &offsets[0], offsets.size(), sizeof(unsigned int), false); 

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

	//unsigned int *offsets1 = (unsigned int *)malloc(num_of_keys*sizeof(unsigned int));
	unsigned int total_size = 0;

	for(int m = 0; m < num_of_keys; m++)
	{
		char *key_address = output_keys + key_index[m];
		char *val_address = output_vals + val_index[m];

		char *key = (char *)key_address;
		//int offset = key.y;
		//offsets1[m] = offset;	
	//	int len = *(int *)val_address;
		total_size+= *(int *)val_address;
		cout<<"line is: "<<key<<endl;
		
		//cout<<filebuf+offset<<": "<<len<<endl;
	}

	cout<<"*******************************************"<<endl;
	cout<<"total size is: "<<total_size<<endl;

	scheduler.destroy();

	/*second pass*/
	cout<<endl;
	cout<<endl;
/*
	cout<<"####Doing mapreduce, second pass####"<<endl;
	*filebuf = 1; //indicate that it is second pass
	Scheduler scheduler1(filebuf, size+1, offsets1, num_of_keys, sizeof(unsigned int), false); 

	double beforemap1 = rtclock();
	scheduler1.do_mapreduce();
	double aftermap1 = rtclock();

	cout<<"The time of mapreduce: "<<aftermap1-beforemap1<<endl;

	unsigned total_size_of_keys1 = scheduler1.get_total_key_size();
	unsigned num_of_keys1 = scheduler1.get_total_key_num();

	struct output output1 = scheduler1.get_output();

	char *output_keys1 = output1.output_keys;
	char *output_vals1 = output1.output_vals;
	unsigned int *key_index1 = output1.key_index;
	unsigned int *val_index1 = output1.val_index;

	cout<<"*******************************************"<<endl;

	for(int m = 0; m < num_of_keys1; m++)
	{
		char *key_address = output_keys1 + key_index1[m];
		char *val_address = output_vals1 + val_index1[m];

		Key key = *(Key *)key_address;
		int offset = key.y;
		int num = *(int *)val_address;
		
		cout<<filebuf+offset<<": "<<num<<endl;
	}

	cout<<"*******************************************"<<endl;

	scheduler1.destroy();
	*/
	//free(offsets1);

	delete[] filebuf;	
}
