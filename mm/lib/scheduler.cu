#include <iostream>

#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "reductionobject_s.cu"
#endif

#ifndef REDUCTIONOBJECTG
#define REDUCTIONOBJECTG
#include "reductionobject_g.cu"
#endif

#ifndef HEADER 
#define HEADER
#include "header.h"
#endif

#ifndef ERRORHANDLING
#define ERRORHANDLING
#include "errorhandling.h"
#endif

#ifndef SCAN
#define SCAN
#include "scan.h"
#endif

using namespace std;
#include <cuda_runtime.h>

#include <stdio.h>

#include <math.h>

#ifndef DS
#define DS
#include "ds.h"
#endif

template <class T1, class T2> 
__device__ void merge(T1 *dstobject, T2 *srcobject)
{
	for(int index = 0; index<srcobject->num_buckets; index++)
	{
		if((srcobject->buckets)[index]!=0)		
		{
			int key_size = srcobject->get_key_size(index);  	
			int value_size = srcobject->get_value_size(index);
			void *key = srcobject->get_key_address(index);	
			void *value = srcobject->get_value_address(index);
			int a = 0;
			dstobject->insert_or_update_value(&a, key, key_size, value, value_size);
		}
	}
}

/*Merge the shared object to the global object*/
__device__ void merge_to_global(Reduction_Object_G *dst, Reduction_Object *src)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int group_size = blockDim.x/NUM_GROUPS;

	//tid%group_size is the id in the group
	for(int index = tid%group_size; index < src->num_buckets; index += group_size)
	{
		if((src->buckets)[index]!=0)	
		{
			int key_size = src->get_key_size(index);
			int value_size = src->get_value_size(index);
			void *key = src->get_key_address(index);
			void *value = src->get_value_address(index);
			int a =0;
			dst->insert_or_update_value(&a, key, key_size, value, value_size);
		}
	}
}

__global__ void start(int *testdata, Reduction_Object_G *object_g, void *global_data, 
void *global_offset, unsigned int offset_number, unsigned int unit_size)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int bid = blockIdx.x;
	const unsigned int num_threads = gridDim.x*blockDim.x;

	object_g->oma_init(testdata);
	__syncthreads();

	#ifdef USE_SHARED

	#ifdef SORT_WHEN_FULL
	__shared__ Reduction_Object objects[2][NUM_GROUPS];
	__shared__ unsigned int use_index; //indicates which objects to use
	__shared__ unsigned int merge_index;
	#endif

	#ifdef MERGE_WHEN_FULL
	__shared__ Reduction_Object objects[NUM_GROUPS];
	#endif

	__shared__ int do_merge;
	__shared__ int finished;
	const unsigned int group_size = blockDim.x/NUM_GROUPS;
	const unsigned int gid = tid/group_size;

	#ifdef MERGE_WHEN_FULL
	objects[gid].oma_init();	
	#endif

	#ifdef SORT_WHEN_FULL
	objects[0][gid].oma_init();
	objects[1][gid].oma_init();
	#endif

	if(tid==0) //The first NUM_GROUPs+1 threads initialize the reduction obejcts
	{
		do_merge = 0;
		finished = 0;
		#ifdef SORT_WHEN_FULL
		use_index = 0; //initially, use the first objects
		merge_index = 1;
		#endif
	}

	__syncthreads();

	bool flag = true;
	int i = bid*blockDim.x+tid;

	while(finished!=NUM_THREADS)
	{

		__syncthreads();


		for(; i < offset_number; i+= num_threads)		
	      	{
			if(do_merge)
			break;

			#ifdef MERGE_WHEN_FULL
			bool success = Map_Reduce::map(testdata, &objects[gid], global_data, ((char *)global_offset+unit_size*i));
			#endif

			#ifdef SORT_WHEN_FULL
			bool success = Map_Reduce::map(testdata, &objects[use_index][gid], global_data, ((char *)global_offset+unit_size*i));
			#endif

			if(!success)
			{
				atomicExch(&do_merge, 1);
				break;
			}
		}	

		if(flag&&i>=offset_number)
		{
			flag = false;
			atomicAdd(&finished, 1);
		}

		__syncthreads();

		/*the action taken after the reduction object is full*/
		#ifdef MERGE_WHEN_FULL
		if(tid%group_size==0)
		{
			merge(object_g, &objects[gid]);
			atomicExch(&do_merge, 0);
		}

		__syncthreads();

		objects[gid].oma_init();
		#endif

		#ifdef SORT_WHEN_FULL
		/*First sort each reduction object*/
		objects[use_index][gid].bitonic_sort(testdata);
		__syncthreads();

		objects[merge_index][gid].merge(&objects[use_index][gid], SORT_REMAIN);

		__syncthreads();

		objects[use_index][gid].oma_init();

		__syncthreads();

		if(gid==0&&tid%group_size==0)
		{
			unsigned int tmp = use_index;
			use_index = merge_index;
			merge_index = tmp;
			atomicExch(&do_merge, 0);
		}
		#endif
	}

	#ifdef SORT_WHEN_FULL
	if(tid%group_size==0)
	merge(object_g, &objects[use_index][gid]);
	#endif

	#else //do not use shared memory object
	for(int i = bid*blockDim.x+tid; i< offset_number; i+= num_threads)
	Map_Reduce::map(testdata, object_g, global_data, ((char *)global_offset+unit_size*i));			
	#endif
} 

__global__ void copy_hash_to_array(Reduction_Object_G *object_g, void *key_array, 
unsigned int *key_start_per_bucket, void *val_array, unsigned int *val_start_per_bucket, 
unsigned int *pair_start_per_bucket, unsigned int *key_index, unsigned int *val_index)
{
	const unsigned int tid = threadIdx.x;		
	const unsigned int bid = blockIdx.x;
	const unsigned int num_threads = gridDim.x*blockDim.x;

	for(int i = bid*blockDim.x+tid; i < object_g->num_buckets; i+=num_threads)
	{
		if((object_g->buckets)[i]!=0)	
		{
			int key_size = object_g->get_key_size(i);	
			int val_size = object_g->get_value_size(i);
			void *key = object_g->get_key_address(i);	
			void *val = object_g->get_value_address(i);
			unsigned int key_array_start = key_start_per_bucket[i];
			unsigned int val_array_start = val_start_per_bucket[i];
			unsigned int offset_pos = pair_start_per_bucket[i];
			copyVal((char *)key_array + key_array_start, key, key_size);
			copyVal((char *)val_array + val_array_start, val, val_size);
			key_index[offset_pos] = key_array_start;
			val_index[offset_pos] = val_array_start;
		}
	}
}

/*trunk the object to just k buckets*/
__global__ void trunk(Reduction_Object_G *object, unsigned int k)
{
	object->num_buckets = k;
}

__global__ void merge_global_object(unsigned int *testdata, Reduction_Object_G *object, unsigned int k, unsigned int j)
{
	object->bitonic_merge(testdata, k, j);
}

struct Scheduler
{
	public:

		void *global_data_h; //Stores the actual data in the host memory
		unsigned int data_size; //the size of the global data
		void *global_offset_h; //Stores the offset information in the host memory, which is used to split task 
		unsigned int offset_number; //number of offsets
		unsigned int unit_size; //the unit_size of offset, used to jump
		unsigned int total_key_num;
		unsigned int total_key_size;
		unsigned int total_value_num;
		unsigned int total_value_size;
		struct output output;
		bool sort;
		Reduction_Object_G *rog;

		//The global_offset contains the position of each data element to be mapped 
		Scheduler(void *global_data, unsigned int data_size, void *global_offset, unsigned int offset_number, unsigned int unit_size, bool sort)
		{
			this->global_data_h = global_data;
			this->data_size = data_size;
			this->global_offset_h = global_offset;
			this->offset_number = offset_number;
			this->unit_size = unit_size;
			this->sort = sort;
		}

		void sort_object()
		{
			int numThreadsSort = 32;		
			int numBlocksSort = (int)ceil((double)NUM_BUCKETS_G/(double)numThreadsSort);
			
			cout<<"numBlocksSort: "<<numBlocksSort<<endl;
			dim3 grid(numBlocksSort, 1, 1);
			dim3 block(numThreadsSort, 1, 1);

			unsigned int *testdata;
			cudaMalloc((void **)&testdata, sizeof(int));

			for(unsigned int k = 2; k <= NUM_BUCKETS_G; k*=2)
				for(unsigned int j = k/2; j>0; j/=2)
				{
					merge_global_object<<<grid, block, 0>>>(testdata, rog, k, j);
					CUT_CHECK_ERROR("merge_global_object");
					cudaThreadSynchronize();
				}

			dim3 grid_getkey(1, 1, 1);
			dim3 block_getkey(1, 1, 1);

			cudaFree(testdata);
		}

		void trunk_object(unsigned int k)
		{
			dim3 grid(1, 1, 1);	
			dim3 block(1, 1, 1);	
			trunk<<<grid, block, 0>>>(rog, k);
		}

		unsigned int get_total_key_num()
		{
			return total_key_num;
		}

		unsigned int get_total_key_size()
		{
			return total_key_size;
		}
	
		struct output get_output()
		{
			return output;
		}

		void do_mapreduce()
		{
			//copy data from host to device, and conduct emitting and reducing		
			void *global_data_d;
			void *global_offset_d;
			cudaMalloc((void **)&global_data_d, data_size);
			cudaMalloc((void **)&global_offset_d, unit_size*offset_number);
			cudaMemcpy(global_data_d, global_data_h, data_size, cudaMemcpyHostToDevice);
			cudaMemcpy(global_offset_d, global_offset_h, unit_size*offset_number, cudaMemcpyHostToDevice);
			
			int *testdata;
			int testdata_h;


			cudaMalloc((void **)&testdata, sizeof(int));
			cudaMemset(testdata, 0, sizeof(int));

			/*first allocate a global reduction object on host*/
			Reduction_Object_G *rogh = (Reduction_Object_G *)malloc(sizeof(Reduction_Object_G));
			memset(rogh, 0, sizeof(Reduction_Object_G));
			//rogh->remaining_buckets = NUM_BUCKETS_G;
			rogh->num_buckets = NUM_BUCKETS_G;
			//rogh->memory_offset= 0;
			/*second copy the object on host to device*/
			cudaMalloc((void **)&rog, sizeof(Reduction_Object_G));
			cudaMemcpy(rog, rogh, sizeof(Reduction_Object_G), cudaMemcpyHostToDevice);

			dim3 grid(NUM_BLOCKS, 1, 1);
			dim3 block(NUM_THREADS, 1, 1);

			double beforemapreduce = rtclock();
			start<<<grid, block, 0>>>(testdata, rog, global_data_d, global_offset_d, offset_number, unit_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("START");
			double aftermapreduce = rtclock();

			cout<<"Map Reduce time: "<<aftermapreduce-beforemapreduce<<endl;
			cout<<"*******************************************"<<endl;

			cudaMemcpy(&testdata_h, testdata, sizeof(int), cudaMemcpyDeviceToHost);
		
			cout<<"Test data: "<<testdata_h<<endl;
			cout<<"*******************************************"<<endl;

			if(sort)
			sort_object();

			/*Next, get the size*/
			unsigned int * key_start_per_bucket;	
			unsigned int * val_start_per_bucket;	
			unsigned int * pair_start_per_bucket;	

			cudaMalloc((void **)&key_start_per_bucket, sizeof(unsigned int)*NUM_BUCKETS_G);
			cudaMalloc((void **)&val_start_per_bucket, sizeof(unsigned int)*NUM_BUCKETS_G);
			cudaMalloc((void **)&pair_start_per_bucket, sizeof(unsigned int)*NUM_BUCKETS_G);

			cudaMemset(key_start_per_bucket, 0, sizeof(unsigned int)*NUM_BUCKETS_G);
			cudaMemset(val_start_per_bucket, 0, sizeof(unsigned int)*NUM_BUCKETS_G);
			cudaMemset(pair_start_per_bucket, 0, sizeof(unsigned int)*NUM_BUCKETS_G);

			unsigned int * key_size_per_bucket = rog->key_size_per_bucket;
			unsigned int * val_size_per_bucket = rog->value_size_per_bucket;
			unsigned int * pairs_per_bucket = rog->pairs_per_bucket;

			total_key_size = prefix_sum(key_size_per_bucket, key_start_per_bucket, NUM_BUCKETS_G);	
			total_value_num = prefix_sum(pairs_per_bucket, pair_start_per_bucket, NUM_BUCKETS_G);
			total_value_size = prefix_sum(val_size_per_bucket, val_start_per_bucket, NUM_BUCKETS_G);
			total_key_num = total_value_num;
			cout<<"total key size: "<<total_key_size<<endl;
			cout<<"total value size: "<<total_value_size<<endl;
			cout<<"total key_num: "<<total_key_num<<endl;
			cout<<"total value_num: "<<total_value_num<<endl;

			/*Next, copy the data from device to host*/
			char *output_keys_d;	
			char *output_vals_d;
			unsigned int *key_index_d;
			unsigned int *val_index_d;

			cudaMalloc((void **)&output_keys_d, total_key_size);
			cudaMalloc((void **)&output_vals_d, total_value_size);
			cudaMalloc((void **)&key_index_d, sizeof(unsigned int)*total_key_num);
			cudaMalloc((void **)&val_index_d, sizeof(unsigned int)*total_value_num);

			copy_hash_to_array<<<grid, block>>>(rog, output_keys_d, key_start_per_bucket, output_vals_d, val_start_per_bucket, pair_start_per_bucket, key_index_d, val_index_d);	
			cudaThreadSynchronize();

			/*Allocate space on host*/
			char *output_keys = (char *)malloc(total_key_size);
			char *output_vals = (char *)malloc(total_value_size);
			unsigned int *key_index = (unsigned int *)malloc(sizeof(unsigned int)*total_key_num);
			unsigned int *val_index = (unsigned int *)malloc(sizeof(unsigned int)*total_value_num);

			cudaMemcpy(output_keys, output_keys_d, total_key_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(output_vals, output_vals_d, total_value_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(key_index, key_index_d, sizeof(unsigned int)*total_key_num, cudaMemcpyDeviceToHost);
			cudaMemcpy(val_index, val_index_d, sizeof(unsigned int)*total_value_num, cudaMemcpyDeviceToHost);

			output.output_keys = output_keys;
			output.output_vals = output_vals;
			output.key_index = key_index;
			output.val_index = val_index;

			cudaFree(key_start_per_bucket);
			cudaFree(val_start_per_bucket);
			cudaFree(pair_start_per_bucket);

			cudaFree(output_keys_d);
			cudaFree(output_vals_d);
			cudaFree(key_index_d);
			cudaFree(val_index_d);
		
			free(rogh);
			cudaFree(global_data_d);
			cudaFree(global_offset_d);
			cudaFree(testdata);
		}

		void destroy()
		{
			free(output.output_keys);
			free(output.output_vals);
			free(output.key_index);
			free(output.val_index);
			cudaFree(rog);
		}
};
