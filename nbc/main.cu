#include <iostream>
using namespace std;
#include <vector>
#include "lib/scheduler.cu"

#ifndef DS_h 
#define DS_h
#include "DS.h"
#endif

#include <getopt.h>

#include "nbc.h"

int main(int argc, char *argv[])
{
	char shortopts[] = "c:t:o:r:s:";
 	struct option longopts[] = 
   	{
         	{"color", 1, NULL, 'c'},
         	{"type", 1, NULL, 't'},
         	{"origin", 1, NULL, 'o'},
		{"transmission", 1, NULL, 'r'},
		{"stolen", 1, NULL, 's'},
         	{0, 0, 0, 0}
     	};

	int opt;
	char color[8] ;
	char type[8] ;
	char origin[8] ;
	char transmission[8] ;
	char stolen[8] ;

	while((opt = getopt_long(argc, argv, shortopts, longopts, NULL)) != -1)
	{
		switch(opt)	
		{
			case 'c':
				strcpy(color, optarg);
				cout<<"Query color: "<<color<<endl;
				break;
			case 't':
				strcpy(type, optarg);
				cout<<"Query type: "<<type<<endl;
				break;
			case 'o':
				strcpy(origin, optarg);
				cout<<"Query origin: "<<origin<<endl;
				break;
			case 'r':
				strcpy(transmission, optarg);
				cout<<"Query transmission: "<<transmission<<endl;
				break;
			case 's':
				strcpy(stolen, optarg);
				cout<<"Query stolen: "<<stolen<<endl;
				break;
		}
	}

	double beforeload = rtclock();
	char filename[] = "dataset";
	void *rawbuf;
	unsigned int size;
	map_file(filename, rawbuf, size);

	if(!rawbuf)
	{
		cout<<"error opening file "<<filename<<endl;
		return 1;
	}

	char *input = new char[5+size];
	input[0] = get_color(color);
	input[1] = get_type(type);
	input[2] = get_origin(origin);
	input[3] = get_transmission(transmission);
	input[4] = get_stolen(stolen);

	for(int i = 0; i<5; i++)
	cout<<(int)input[i]<<endl;

	char *data = input+5;
	memcpy(data, rawbuf, size);

	vector<int> offsets;
	offsets.push_back(5);

	for(int i = 0; i<size; i++)	
	{
		char tmp =data[i];
		if(tmp=='\t')
			data[i] = '\0';
		if(tmp=='\n')
		{
			data[i] = '\0';
			if(i+1<size)
			offsets.push_back(i+1+5);
		}
	}

	double afterload = rtclock();

	cout<<"Load time: "<<afterload-beforeload<<endl;

	cout<<"The number of offsets: "<<offsets.size()<<endl;

	cout<<"Doing mapreduce..."<<endl;

	Scheduler scheduler(input, 5+size, &offsets[0], offsets.size(), sizeof(unsigned int), false); 
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

	for(int m = 0; m < num_of_keys; m++)
	{
		char *key_address = output_keys + key_index[m];
		char *val_address = output_vals + val_index[m];

		int key = *(int *)key_address;
		int val = *(int *)val_address;
				
		cout<<key<<": "<<val<<endl;
	}

	cout<<"*******************************************"<<endl;

	int numbers[30];
	for(int i = 0; i<30; i++)
	numbers[i] = 0;
	for(int i = 0; i<num_of_keys; i++)
	{
		char *key_address = output_keys + key_index[i];
		char *val_address = output_vals + val_index[i];

		int key = *(int *)key_address;
		int val = *(int *)val_address;
		numbers[key] = val;
	}

	double p_age[5];
	for(int h = 0; h<5; h++)
		p_age[h]=(double)numbers[25+h]/offsets.size(); 
	
	double p[25];

	for(int k = 0; k<5; k++)
	{
		for(int x = 0; x<5; x++)
		{
			p[k*5 + x] = 0;
			if(numbers[25+k]!=0)
			p[k*5 + x] = (double)numbers[k*5 + x]/(double)numbers[25+k];	
		}
	}

	double ps[5];

	for(int y = 0; y<5; y++)
	{
		ps[y] = 1;
		for(int z = 0; z<5;z++)
		ps[y]*=p[y*5+z];
		ps[y]*=p_age[y];
	}

	double max_p = ps[0];
	unsigned int max_index = 0;

	for(int l = 1; l<5; l++)
	{
		if(ps[l]>max_p)
		{
			max_p = ps[l];
			max_index = l;
		}
	}

	cout<<"The age of the car is: "<<max_index+1<<endl;
	cout<<"And the max possibility is: "<<max_p<<endl;

	for(int n = 0; n < 5; n++)
		cout<<ps[n]<<endl;

	scheduler.destroy();
	delete[] input;
}
