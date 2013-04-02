#ifndef COMMON
#define COMMON
#include "common.h"
#endif

struct Reduction_Object_G
{
	public:
		unsigned int num_buckets;
		int remaining_buckets;
		int locks[NUM_BUCKETS_G];
		unsigned long long int buckets[NUM_BUCKETS_G]; //every bucket contains two indexes

		unsigned int memory_pool[MAX_POOL_IN_OBJECT_G];

		unsigned int memory_offset;

		__device__ void oma_init();
		/*returns the index*/
		__device__ int omalloc(unsigned int size);
		__device__ void *oget_address(unsigned int index);

		__device__ bool insert_or_update_value(int *testdata, void *key, unsigned short key_size, void *value, unsigned short value_size);

		__device__ void * get_key_address(unsigned int bucket_index);
		__device__ unsigned short get_key_size(unsigned int bucket_index);

		__device__ void * get_value_address(unsigned int bucket_index);
		__device__ int get_compare_value(unsigned int bucket_index1, unsigned int bucket_index2);
		__device__ void swap(unsigned long long int &a, unsigned long long int &b);
		__device__ void bitonic_merge(unsigned int *testdata, unsigned int k, unsigned int j);
};

//__device__ inline Reduction_Object_G* newReduction_Object_G();
