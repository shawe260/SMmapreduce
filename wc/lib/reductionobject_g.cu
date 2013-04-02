#ifndef REDUCTIONOBJECTGH
#define REDUCTIONOBJECTGH
#include "reductionobject_g.h"
#endif

#ifndef ATOMICUTIL
#define ATOMICUTIL
#include "atomicutil.h"
#endif

#ifndef MAPREDUCE
#define MAPREDUCE
#include "../mapreduce.h"
#endif

#ifndef UTIL
#define UTIL
#include "util.h"
#endif

#ifndef COMMON
#define COMMON
#include "common.h"
#endif

#ifndef HASH
#define HASH
#include "hash.h"
#endif

#ifndef HEADER
#define HEADER
#include "header.h"
#endif

__shared__ volatile unsigned int global_object_offset[NUM_THREADS/WARP_SIZE];

__device__ void Reduction_Object_G::oma_init()
{
	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int num_groups_g = NUM_THREADS/WARP_SIZE;
	unsigned int gid_g = tid/WARP_SIZE;
	if(tid%WARP_SIZE==0)
	global_object_offset[gid_g] = MAX_POOL_IN_OBJECT_G*(gid_g + num_groups_g*bid)/(num_groups_g*NUM_BLOCKS);
}
//__device__ void Reduction_Object_G::oma_init()
//{
//	/*set all buckets to 0*/
//	int i = 0;
//	for(i = 0; i<NUM_BUCKETS_G; i++)
//	{
//		buckets[i] = 0;
//		locks[i] = 0;
//	}
//
//	for(i = 0; i<MAX_POOL_IN_OBJECT_G; i++)
//	memory_pool[i] = 0;
//
//	remaining_buckets = NUM_BUCKETS_G;
//	num_buckets = NUM_BUCKETS_G;
//	memory_offset = 0;
//}

__device__ int Reduction_Object_G::omalloc(unsigned int size)
{
	size = align(size)/ALIGN_SIZE;

	//unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	//unsigned int num_groups_g = NUM_THREADS/WARP_SIZE;
	unsigned int gid_g = tid/WARP_SIZE;


	unsigned int offset = atomicAdd((unsigned int *)&global_object_offset[gid_g], size);

//	if(offset+size>MAX_POOL_IN_OBJECT_G)
//	return -1; //-1 stands for fail
//
//	else
	return offset; 
}

__device__ void * Reduction_Object_G::oget_address(unsigned int index)
{
	return memory_pool + index;	
}

__device__ bool Reduction_Object_G::insert_or_update_value(int *testdata, void *key, unsigned short key_size, void *value, unsigned short value_size)
{

	//No buckets available
//	if(remaining_buckets==0)
//	return false;

	//First, calculate the bucket index number
	unsigned int h = hash(key, key_size);	

//	if(blockDim.x*blockIdx.x+threadIdx.x==0)
//	atomicAdd(testdata, *(unsigned int *)key);

	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;
	unsigned long long int kvn = 0;
	
	bool DoWork = true;
	bool ret = true;
	int stride = 1;

	//lock only when new key is to be inserted
	while(true)
	{

		DoWork = true;
		while(DoWork)
		{
			//Second, test whether the bucket is empty
			if(getLock(&locks[index]))
			{
				if(buckets[index]==0)
				{
					//If the bucket is empty, the key has not appeared in the reduction object, create a new key-value pair
					//Copy the real data of key and value to shared memory
					int k  = omalloc(2+key_size);//The first byte stores size of the key, and the second byte stores the size of the val
					if(k==-1)//The shared memory pool is full
					ret = false;
							
					int v = omalloc(value_size);
					if(v==-1)
					ret = false;	
	
					//store the key index and value index to the temparary variable 
	
					copyVal((int *)&kvn, &k, sizeof(k));
					copyVal((int *)&kvn + 1, &v, sizeof(v));
						
	
					char *key_size_address = (char *)oget_address(k);
					char *value_size_address = key_size_address + 1;
					*key_size_address = key_size; 
					*value_size_address = value_size;
					
					//The start address of the key data
					void *key_data_start = key_size_address + 2;
					//The start address of the value data
					void *value_data_start = oget_address(v); 
			
					//Copy the key data to shared memory
					copyVal(key_data_start,key,key_size);
					//Copy the value data to shared memory
					copyVal(value_data_start,value,value_size);
					buckets[index] = kvn;
	
					key_size_per_bucket[index] = key_size;	
					value_size_per_bucket[index] = value_size;
					pairs_per_bucket[index] = 1;
	
					//atomicAdd(&remaining_buckets, -1);
	
					finish = 1;
					DoWork = false;
					releaseLock(&locks[index]);
				}
	
				else 
				{
					unsigned short size = get_key_size(index);
					void *key_data = get_key_address(index);
		
					if(Map_Reduce::equal(key_data, size, key, key_size ))
					{
						Map_Reduce::reduce(get_value_address(index), get_value_size(index), value, value_size);
						DoWork = false;
						finish = 1;
						ret = true;
						releaseLock(&locks[index]);
					}
	
					else
					{
						DoWork = false;
						releaseLock(&locks[index]);
						index = (index+stride)%NUM_BUCKETS_G;
					}
				}
			}
		}
		if(finish)
		return ret;
	}
}

__device__ void * Reduction_Object_G::get_key_address(unsigned int bucket_index)
{
	if(buckets[bucket_index]==0)
	return 0;

	unsigned int key_index = ((unsigned int *)&buckets[bucket_index])[0]; 
	char *key_size_address = (char *)oget_address(key_index);
	return key_size_address + 2; //key data starts after the two size byte
}

__device__ unsigned int Reduction_Object_G::get_key_size(unsigned int bucket_index)
{
	if(buckets[bucket_index]==0)
	return 0;

	unsigned int key_index = ((unsigned int *)&buckets[bucket_index])[0]; 
	return *(char *)oget_address(key_index);
}

__device__ void * Reduction_Object_G::get_value_address(unsigned int bucket_index)
{
	if(buckets[bucket_index]==0)
	return 0;

	unsigned int value_index = ((unsigned int *)&buckets[bucket_index])[1]; 
	return oget_address(value_index);
}

__device__ unsigned int Reduction_Object_G::get_value_size(unsigned int bucket_index)
{
	unsigned int key_index = ((unsigned int *)&buckets[bucket_index])[0]; 
	return *((char *)oget_address(key_index) + 1);
}

__device__ int Reduction_Object_G::get_compare_value(unsigned int bucket_index1, unsigned int bucket_index2)
{
	unsigned long long int bucket1 = buckets[bucket_index1];
	unsigned long long int bucket2 = buckets[bucket_index2];
	int compare_value = 0;

	if(bucket1==0&&bucket2==0)
	compare_value = 0;

	else if(bucket2==0)
	compare_value = -1;

	else if(bucket1==0)
	compare_value = 1;

	else
	{
		unsigned short key_size1 = get_key_size(bucket_index1);	
		unsigned short key_size2 = get_key_size(bucket_index2);	
		void *key_addr1 = get_key_address(bucket_index1);
		void *key_addr2 = get_key_address(bucket_index2);
		unsigned short value_size1 = get_value_size(bucket_index1);
		unsigned short value_size2 = get_value_size(bucket_index2);
		void *value_addr1 = get_value_address(bucket_index1);
		void *value_addr2 = get_value_address(bucket_index2);

		compare_value = Map_Reduce::compare(key_addr1, key_size1, key_addr2, key_size2, value_addr1, value_size1, value_addr2, value_size2);
	}

	return compare_value;
}

__device__ void Reduction_Object_G::swap(unsigned long long int &a, unsigned long long int &b)
{
	unsigned long long int tmp = a;
	a = b;
	b = tmp;
}

__device__ void swap_int(unsigned int &a, unsigned int &b)
{
	unsigned int tmp =a;	
	a = b;
	b = tmp;
}

/*The sort is based on the key value*/
__device__ void Reduction_Object_G::bitonic_merge(unsigned int *testdata, unsigned int k, unsigned int j)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;
	const unsigned int id = bid*blockDim.x+tid;
	const unsigned int ixj = id ^ j;//j controls which thread to use 

	int compare = 0;
	if(ixj<NUM_BUCKETS_G && id<NUM_BUCKETS_G)
	if(ixj > id)
	{
		compare = get_compare_value(id, ixj);
		if((id & k) == 0)//k controls the direction
		{
			if(compare>0)
			{
				swap(buckets[id], buckets[ixj]);
				swap_int(key_size_per_bucket[id], key_size_per_bucket[ixj]);
				swap_int(value_size_per_bucket[id], value_size_per_bucket[ixj]);
				swap_int(pairs_per_bucket[id], pairs_per_bucket[ixj]);
			}
		}

		else
		{
			if(compare<0)
			{
				swap(buckets[id], buckets[ixj]);
				swap_int(key_size_per_bucket[id], key_size_per_bucket[ixj]);
				swap_int(value_size_per_bucket[id], value_size_per_bucket[ixj]);
				swap_int(pairs_per_bucket[id], pairs_per_bucket[ixj]);
			}
		}
	}
}
