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
#include "mapreduce.h"
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

__device__ void Reduction_Object_G::oma_init()
{
	/*set all buckets to 0*/
	int i = 0;
	for(i = 0; i<NUM_BUCKETS_G; i++)
	{
		buckets[i] = 0;
		locks[i] = 0;
	}

	for(i = 0; i<MAX_POOL_IN_OBJECT_G; i++)
	memory_pool[i] = 0;

	remaining_buckets = NUM_BUCKETS_G;
	num_buckets = NUM_BUCKETS_G;
	memory_offset = 0;
}

__device__ int Reduction_Object_G::omalloc(unsigned int size)
{
	size = align(size)/ALIGN_SIZE;
	unsigned int offset = atomicAdd(&memory_offset, size);

	if(offset+size>MAX_POOL_IN_OBJECT_G)
	return -1; //-1 stands for fail

	else
	return offset; 
}

__device__ void * Reduction_Object_G::oget_address(unsigned int index)
{
	return memory_pool + index;	
}

__device__ bool Reduction_Object_G::insert_or_update_value(int *testdata, void *key, unsigned short key_size, void *value, unsigned short value_size)
{

	//atomicAdd(testdata, 1);
	//No buckets available
	if(remaining_buckets==0)
	return false;

	//First, calculate the bucket index number
	unsigned int h = hash(key, key_size);	
	unsigned int index = h%NUM_BUCKETS_G;
	unsigned int finish = 0;
	unsigned long long int kvn = 0;
	
	bool DoWork = true;
	bool ret = true;
	int stride = NUM_BUCKETS_G;

	//lock only when new key is to be inserted

	if(buckets[index]==0)
	while(DoWork)
	{
		//Second, test whether the bucket is empty
		if(buckets[index]!=0)
		break;
		if(getLock(&locks[index]))
		{
			if(buckets[index]==0)
			{
				//If the bucket is empty, the key has not appeared in the reduction object, create a new key-value pair
				//Copy the real data of key and value to shared memory
				int k  = omalloc(1+key_size);//The first byte stores size of the key
				if(k==-1)//The shared memory pool is full
				ret = false;
						
				int v = omalloc(value_size);
				if(v==-1)
				ret = false;	

				//store the key index and value index to the temparary variable 
			//	*((unsigned int *)&kvn)= k;
			//	*(1+(unsigned int *)&kvn)= v;

				copyVal((int *)&kvn, &k, sizeof(k));
				copyVal((int *)&kvn + 1, &v, sizeof(v));
					
				//*testdata = ((unsigned short *)&kvn)[1];
				//atomicAdd(testdata, 1);

				char *key_size_address = (char *)oget_address(k);
				*key_size_address = key_size; 
				//copy(&kvn1, &kvn, sizeof(unsigned int));
				
				//The start address of the key data
				void *key_data_start = key_size_address + 1;
				//The start address of the value data
				void *value_data_start = oget_address(v); 
		
				//Copy the key data to shared memory
				copyVal(key_data_start,key,key_size);
				//Copy the value data to shared memory
				copyVal(value_data_start,value,value_size);
				//If the bucket is still empty, insert the information of the new key-value pair into it	
				//CAS64(&buckets[index], 0, kvn);	
				atomicAdd(&buckets[index], kvn);

				atomicAdd(&remaining_buckets, -1);
				finish = 1;
			}
			releaseLock(&locks[index]);
			DoWork = false;
		}
	}

	if(finish)
	return ret;
	//The bucket is not empty, test its key first, and if not equal, test the next bucket 
	while(1)
	{
		DoWork = true;
		if(remaining_buckets==0)
		return false;

		//The keys are equal
		unsigned short size = get_key_size(index);
		void *key_data = get_key_address(index);
		if(equal(key_data, size, key, key_size ))
		{
			//*testdata=*testdata+1;
			//reduce the key to the reduction object
			//atomicAdd((int *)get_value_address(index), *(int *)value);	
			//getLock(&lockVal);
			while(DoWork)
			{	
				if(getLock(&locks[index]))
				{
			//		atomicAdd(testdata, 1);
					Map_Reduce::reduce(get_value_address(index), value);
					releaseLock(&locks[index]);
					DoWork = false;
					finish = 1;
					ret = true;
				}
			}
		}

		DoWork = true;
		if(finish)
		return ret;

		index = (index+stride)%NUM_BUCKETS_G;
		if(stride!=1)
		stride/=2;

		//If the next bucket is empty
		//Only when inserting, use lock
		if(buckets[index]==0)
		while(DoWork)
		{
			if(buckets[index]!=0)
			break;
			if(getLock(&locks[index]))
			{
				//If the key and value has not been allocated
				if(buckets[index]==0)
				{
					if(kvn==0)	
					{
						//If the bucket is empty, the key has not appeared in the reduction object, create a new key-value pair
						//Copy the real data of key and value to shared memory
						int k  = omalloc(1+key_size);//The first byte stores size of the key
						if(k==-1)//The shared memory pool is full
						ret = false;
								
						int v = omalloc(value_size);
						if(v==-1)
						ret = false;	

						*((unsigned int *)&kvn)= k;
						*(1+(unsigned int *)&kvn)= v;

					//	copyVal((int *)&kvn, &k, sizeof(k));
					//	copyVal((int *)&kvn + 1, &v, sizeof(v));

						char *key_size_address = (char *)oget_address(k);
						*key_size_address = key_size; 
						//copy(&kvn1, &kvn, sizeof(unsigned int));
						
						//The start address of the key data
						void *key_data_start = key_size_address+1;
						//The start address of the value data
						void *value_data_start = oget_address(v); 
		
						//Copy the key data to shared memory
						copyVal(key_data_start,key,key_size);
						//Copy the value data to shared memory
						copyVal(value_data_start,value,value_size);
						//If the bucket is still empty, insert the information of the new key-value pair into it	
					}
					//CAS64(&buckets[index], 0, kvn);	
					atomicAdd(&buckets[index], kvn);
					atomicAdd(&remaining_buckets, -1);
					finish = 1;
				}
				releaseLock(&locks[index]);
				DoWork = false;
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
	return key_size_address + 1; //key data starts after the size byte
}

__device__ unsigned short Reduction_Object_G::get_key_size(unsigned int bucket_index)
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
		void *value_addr1 = get_value_address(bucket_index1);
		void *value_addr2 = get_value_address(bucket_index2);

		compare_value = Map_Reduce::compare(key_addr1, key_size1, key_addr2, key_size2, value_addr1, value_addr2);
	}

	return compare_value;
}

__device__ void Reduction_Object_G::swap(unsigned long long int &a, unsigned long long int &b)
{
	unsigned long long int tmp = a;
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
			}
		}

		else
		{
			if(compare<0)
			{
				swap(buckets[id], buckets[ixj]);
			}
		}
	}
}
