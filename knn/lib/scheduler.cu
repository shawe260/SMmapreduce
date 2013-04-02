#include <iostream>

#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "reductionobject_s.cu"
#endif

#ifndef REDUCTIONOBJECTG
#define REDUCTIONOBJECTG
#include "reductionobject_g.cu"
#endif

#ifndef MAPREDUCE
#define MAPREDUCE
#include "mapreduce.h"
#endif

#ifndef HEADER 
#define HEADER
#include "header.h"
#endif

#ifndef ERRORHANDLING
#define ERRORHANDLING
#include "errorhandling.h"
#endif

using namespace std;
#include <cuda_runtime.h>

#include <stdio.h>

#include <math.h>

template <class T1, class T2> 
__device__ void merge(T1 *dstobject, T2 *srcobject, unsigned int value_size)
{
	for(int index = 0; index<srcobject->num_buckets; index++)
	{
		if((srcobject->buckets)[index]!=0)		
		{
			int key_size = srcobject->get_key_size(index);  	
			void *key = srcobject->get_key_address(index);	
			void *value = srcobject->get_value_address(index);
			int a =0;
			dstobject->insert_or_update_value(&a, key, key_size, value, value_size);
		}
	}
}

/*Merge the shared object to the global object*/
__device__ void merge_to_global(Reduction_Object_G *dst, Reduction_Object *src, unsigned int value_size)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int group_size = blockDim.x/NUM_GROUPS;

	//tid%group_size is the id in the group
	for(int index = tid%group_size; index < src->num_buckets; index += group_size)
	{
		if((src->buckets)[index]!=0)	
		{
			int key_size = src->get_key_size(index);
			void *key = src->get_key_address(index);
			void *value = src->get_value_address(index);
			int a =0;
			dst->insert_or_update_value(&a, key, key_size, value, value_size);
		}
	}
}

__global__ void start(int *testarray, int *testdata, Reduction_Object_G *object_g, void *global_data, 
void *global_offset, unsigned int offset_number, unsigned int unit_size, unsigned int value_size)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int bid = blockIdx.x;
	const unsigned int num_threads = gridDim.x*blockDim.x;

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
		*testdata = 0;
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
		merge_to_global(&objects[gid], value_size);
		if(tid%group_size==0)
		{
			//merge(object_g, &objects[gid], value_size);
			atomicExch(&do_merge, 0);
		}

		__syncthreads();

		objects[gid].oma_init();
		#endif

		#ifdef SORT_WHEN_FULL
		/*First sort each reduction object*/
		objects[use_index][gid].bitonic_sort(testdata);
		__syncthreads();

		/*Then insert the first K elements to the merge reduction object*/
	//	if(tid%group_size==0)
	//	merge_index = use_index ^ 1;	
	//	__syncthreads();


//
//		__syncthreads();
		objects[merge_index][gid].merge(&objects[use_index][gid], SORT_REMAIN, value_size);

		__syncthreads();

		objects[use_index][gid].oma_init();

		//__syncthreads();

	//	return;

		__syncthreads();

		if(gid==0&&tid%group_size==0)
		{
			unsigned int tmp = use_index;
			use_index = merge_index;
			merge_index = tmp;
			atomicExch(&do_merge, 0);
		}

		//__syncthreads();

//		if(gid==0)
//		if(tid%group_size==0)
//		objects[use_index][gid].get_used_buckets(testdata);
		//*testdata = objects[use_index][gid].remaining_buckets;
		
//		return;

		#endif
	}

	#ifdef SORT_WHEN_FULL
	if(tid%group_size==0)
	merge(object_g, &objects[use_index][gid], value_size);
	#endif

	#else //do not use shared memory object
	for(int i = bid*blockDim.x+tid; i< offset_number; i+= num_threads)
	Map_Reduce::map(testdata, object_g, global_data, ((char *)global_offset+unit_size*i));			
	#endif
} 

/*Calculate the size info of the reduction object. The size info will be used to copy data from the obejct
* total_size_of_keys: the number of bytes that all the keys will take
* num_of_keys: the total number of different keys
*/
__global__ void get_size_info(Reduction_Object_G * object_g, unsigned int *total_size_of_keys, unsigned int *num_of_keys)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int bid = blockIdx.x;

	for(int index = bid*blockDim.x+tid; index < object_g->num_buckets/*NUM_BUCKETS_G*/; index+=gridDim.x*blockDim.x)
	{
		if((object_g->buckets)[index]!=0)
		{
			atomicAdd(num_of_keys, 1);	
			int key_size = object_g->get_key_size(index);
			atomicAdd(total_size_of_keys, key_size);
		}
	}
}

/*copies data from global reduction object to arrays. We need three arrays, 
* 	key_array: constains the information of the key
*	key_offset_array: contains the offset information of the key_array
*	value_array: contains keys
*	value_size: the size of each value
*	in our framework, we assume that the size of the value is fixed
*/
__global__ void copy_to_array(Reduction_Object_G * object_g, void *key_array, 
int *key_offset_array, void *value_array, int value_size)
{
	const unsigned int tid = threadIdx.x;	
	const unsigned int bid = blockIdx.x;
	
	if(bid==0&&tid==0)
	{
		int key_array_ptr = 0;
		int key_offset_array_ptr = 0;
		int value_array_ptr = 0;
	
		for(int index = 0; index < object_g->num_buckets/*NUM_BUCKETS_G*/; index++)
		{
			if((object_g->buckets)[index]!=0)	
			{
				int key_size = object_g->get_key_size(index);  	
				void *key = object_g->get_key_address(index);	
				void *value = object_g->get_value_address(index);
				key_offset_array[key_offset_array_ptr] = key_array_ptr;
				copyVal((char *)key_array+key_array_ptr, key, key_size);
				copyVal((char *)value_array+value_array_ptr, value, value_size);
	
				key_array_ptr+=key_size;
				key_offset_array_ptr++;
				value_array_ptr+=value_size;
			}
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

/*used for testing*/
__global__ void get_key(Reduction_Object_G *object, unsigned int bucket_index, unsigned int *testdata)
{
	void *key_address = object->get_key_address(bucket_index);	
	copyVal(testdata, key_address, sizeof(unsigned int));
}

struct Scheduler
{
	public:

		void *global_data_h; //Stores the actual data in the host memory
		unsigned int data_size; //the size of the global data
		void *global_offset_h; //Stores the offset information in the host memory, which is used to split task 
		unsigned int offset_number; //number of offsets
		unsigned int unit_size; //the unit_size of offset, used to jump
		unsigned int value_size; //the size of the value

		Reduction_Object_G *rog;
		//The global_offset contains the position of each data element to be mapped 
		Scheduler(void *global_data, unsigned int data_size, void *global_offset, unsigned int offset_number, unsigned int unit_size,  unsigned int value_size)	
		{
			this->global_data_h = global_data;
			this->data_size = data_size;
			this->global_offset_h = global_offset;
			this->offset_number = offset_number;
			this->unit_size = unit_size;
			this->value_size = value_size;
		}

		void sort_object()
		{
			int numThreadsSort = 32;		
			int numBlocksSort = (int)ceil((double)NUM_BUCKETS_G/(double)numThreadsSort);
			
			cout<<"numBlocksSort: "<<numBlocksSort<<endl;
			dim3 grid(numBlocksSort, 1, 1);
			dim3 block(numThreadsSort, 1, 1);

			unsigned int *testdata;
			unsigned int testdata_h;
			cudaMalloc((void **)&testdata, sizeof(int));

			for(unsigned int k = 2; k <= NUM_BUCKETS_G; k*=2)
				for(unsigned int j = k/2; j>0; j/=2)
				{
					merge_global_object<<<grid, block, 0>>>(testdata, rog, k, j);
					CUT_CHECK_ERROR("merge_global_object");
					cudaThreadSynchronize();
					cudaMemcpy(&testdata_h, testdata, sizeof(int), cudaMemcpyDeviceToHost);
				}

			dim3 grid_getkey(1, 1, 1);
			dim3 block_getkey(1, 1, 1);

		//	for(int i = 0; i<10; i++)
		//	{
		//	get_key<<<grid_getkey, block_getkey>>>(rog, i, testdata);
		//	cudaMemcpy(&testdata_h, testdata, sizeof(int), cudaMemcpyDeviceToHost);

		//	cout<<"Key in bucket "<<i<<": "<<testdata_h<<endl;
		//	}

			cudaFree(testdata);
		}

		void trunk_object(unsigned int k)
		{
			dim3 grid(1, 1, 1);	
			dim3 block(1, 1, 1);	
			trunk<<<grid, block, 0>>>(rog, k);
		}

		void get_sizes(unsigned *total_size_of_keys, unsigned *num_of_keys)
		{
			dim3 grid(NUM_BLOCKS, 1, 1);
			dim3 block(NUM_THREADS, 1, 1);

			unsigned int *total_size_of_keys_d;
			unsigned int *num_of_keys_d;

			cudaMalloc((void **)&total_size_of_keys_d, sizeof(unsigned int));
			cudaMalloc((void **)&num_of_keys_d, sizeof(unsigned int));

			cudaMemset(total_size_of_keys_d, 0, sizeof(unsigned int));
			cudaMemset(num_of_keys_d, 0, sizeof(unsigned int));
			
			get_size_info<<<grid, block, 0>>>(rog, total_size_of_keys_d, num_of_keys_d);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("GET_SIZE_INFO");

			cudaMemcpy(total_size_of_keys, total_size_of_keys_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaMemcpy(num_of_keys, num_of_keys_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

			cudaFree(total_size_of_keys_d);
			cudaFree(num_of_keys_d);
		}

		void copy_data(void *key_array_h, int *key_offset_array_h, void *value_array_h, 
		unsigned key_array_size, unsigned num_of_keys, unsigned value_array_size)
		{
			dim3 grid_copy(1, 1, 1);
			dim3 block_copy(1, 1, 1);

			void *key_array;
			int *key_offset_array;
			void *value_array;

			cudaMalloc((void **)&key_array, key_array_size);
			cudaMalloc((void **)&key_offset_array, sizeof(int)*num_of_keys);
			cudaMalloc((void **)&value_array, value_array_size);

			cudaMemset(key_array, 0, key_array_size);
			cudaMemset(key_offset_array, 0, sizeof(int)*num_of_keys);
			cudaMemset(value_array, 0, value_array_size);

			copy_to_array<<<grid_copy, block_copy>>>(rog, key_array, key_offset_array, value_array, value_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("COPY_TO_ARRAY");

			cudaMemcpy(key_array_h, key_array, key_array_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(key_offset_array_h, key_offset_array, sizeof(int)*num_of_keys, cudaMemcpyDeviceToHost);
			cudaMemcpy(value_array_h, value_array, value_array_size, cudaMemcpyDeviceToHost);

			cudaFree(key_array);
			cudaFree(key_offset_array);
			cudaFree(value_array);
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

			int *testarray;
			int testarray_h[NUM_BUCKETS];

			cudaMalloc((void **)&testdata, sizeof(int));
			cudaMalloc((void **)&testarray, sizeof(int)*NUM_BUCKETS);

			/*first allocate a global reduction object on host*/
			Reduction_Object_G *rogh = (Reduction_Object_G *)malloc(sizeof(Reduction_Object_G));
			memset(rogh, 0, sizeof(Reduction_Object_G));
			rogh->remaining_buckets = NUM_BUCKETS_G;
			rogh->num_buckets = NUM_BUCKETS_G;
			rogh->memory_offset= 0;
			/*second copy the object on host to device*/
			cudaMalloc((void **)&rog, sizeof(Reduction_Object_G));
			cudaMemcpy(rog, rogh, sizeof(Reduction_Object_G), cudaMemcpyHostToDevice);

			dim3 grid(NUM_BLOCKS, 1, 1);
			dim3 block(NUM_THREADS, 1, 1);

			double beforemapreduce = rtclock();
			start<<<grid, block, 0>>>(testarray, testdata, rog, global_data_d, global_offset_d, offset_number, unit_size, value_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("START");
			double aftermapreduce = rtclock();

			cout<<"Map Reduce time: "<<aftermapreduce-beforemapreduce<<endl;
			cout<<"*******************************************"<<endl;

			cudaMemcpy(&testdata_h, testdata, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&testarray_h, testarray, sizeof(int)*NUM_BUCKETS, cudaMemcpyDeviceToHost);
		
			cout<<"Test data: "<<testdata_h<<endl;
			cout<<"*******************************************"<<endl;
	
		//	for(int n = 0; n<NUM_BUCKETS; n++)
		//	cout<<"Test array "<<n<<": "<<testarray_h[n]<<endl;
		//	cout<<"*******************************************"<<endl;

			free(rogh);
			cudaFree(global_data_d);
			cudaFree(global_offset_d);
			cudaFree(testdata);
			cudaFree(testarray);
		}

		void destroy()
		{
			cudaFree(rog);
		}
};
