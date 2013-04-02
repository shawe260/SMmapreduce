#include <iostream>
#include "lib/scheduler.cu"


#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <getopt.h>

using namespace std;

void map_file(const char * filename, void * & buf, unsigned int & size){
	int fp=open(filename,O_RDONLY);
	if(fp){
		struct stat filestat;
		fstat(fp, &filestat);
		size=filestat.st_size;
		buf=mmap(0,size,PROT_READ,MAP_PRIVATE,fp,0);
	}
	else
		buf=0;
}

int main(int argc, char *argv[])
{
	struct option longopts[] = 
	{
		{"file", 1, NULL, 'f'},
		{0, 0, 0, 0}
	};

	int opt;
	char *filename;
	opt = getopt_long(argc, argv, "f:", longopts, NULL);

	if(opt=='f')
	{
		filename = optarg;
		cout<<"Using file: "<<filename<<endl;
	}

	else
	{
		cout<<"usage: "<<argv[0]<<" --file filename, or "<<argv[0]<<" -f filename"<<endl;
		return 1;
	}
	
	void * rawbuf;
	unsigned int size;
	map_file(filename, rawbuf, size);
	if(!rawbuf){
		cout<<"error opening file "<<filename<<endl;
		return 1;
	}
	char * filebuf=new char[size+1];
	memcpy(filebuf,rawbuf,size); //copy the file content to the file buffer
	filebuf[size]='\0';

	
	vector<int> myofsts;
	unsigned int myofst=0;
	FILE *fp = fopen(filename, "r");
	
	bool in_word = false;
	char ch;
	do
	{
		ch = fgetc(fp);
		if(ch==EOF)
		break;

		if(ch>='a'&&ch<='z'
		||ch>='0'&&ch<='9'
		||ch>='A'&&ch<='Z')
		{
			if(!in_word)
			{
				in_word = true;
				myofsts.push_back(myofst);
			}
		}

		else
		{
			if(in_word)
			{
				in_word = false;
				filebuf[myofst] = '\0'; //substitute the endl to '\0'
			}
		}

		myofst++;

	}while(ch!=EOF);

//	for(int i =0; i < myofsts.size(); i++)
//	cout<<&filebuf[myofsts[i]]<<endl;

	Scheduler scheduler(filebuf, size+1, &myofsts[0], myofsts.size(), sizeof(int), false); 

	cout<<"The size of input data: "<<size+1<<endl;
	cout<<"The number of offsets: "<<myofsts.size()<<endl;
	scheduler.do_mapreduce();

	unsigned total_size_of_keys = scheduler.get_total_key_size();
	unsigned num_of_keys = scheduler.get_total_key_num();

//	scheduler.get_sizes(&total_size_of_keys, &num_of_keys);

	cout<<"Total size of keys: "<<total_size_of_keys<<endl;
	cout<<"Total number of keys: "<<num_of_keys<<endl;
	cout<<endl;

//	void *key_array_h = (void *)malloc(total_size_of_keys);
//	int *key_offset_array_h = (int *)malloc(num_of_keys*sizeof(int));
//	void *value_array_h = (void *)malloc(sizeof(float)*4*num_of_keys);

//	scheduler.copy_data(key_array_h, key_offset_array_h, value_array_h, total_size_of_keys, num_of_keys, sizeof(float)*4*num_of_keys);

	struct output output = scheduler.get_output();

	char *output_keys = output.output_keys;
	char *output_vals = output.output_vals;
	unsigned int *key_index = output.key_index;
	unsigned int *val_index = output.val_index;

	cout<<"*******************************************"<<endl;

	int total_num = 0;
	for(int m = 0; m<num_of_keys; m++)
	{
		char *key_address = output_keys + key_index[m];	
		char *val_address = output_vals + val_index[m];
		int number = *(int *)val_address;
		cout<<key_address<<": "<<number<<endl;
		total_num+=number;
	}

//	int total_num_words = 0;
//	for(int i = 0; i<num_of_keys; i++)
//	{
//		cout<<(char *)key_array_h+key_offset_array_h[i]<<": "<<((int *)value_array_h)[i]<<endl;
//		total_num_words += ((int *)value_array_h)[i];
//	}
	cout<<endl;
	cout<<"Total number of words: "<<total_num<<endl;

//	free(key_array_h);
//	free(key_offset_array_h);
//	free(value_array_h);
	
	scheduler.destroy();
	delete[] filebuf;
}
