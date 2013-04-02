#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#ifndef	MM 
#define MM 
#include "mm.h"
#endif

#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "lib/reductionobject_s.cu"
#endif

typedef struct Reduction_Object RO;

namespace Map_Reduce
{
	//Define the way of emitting key-value pairs
	template <class T>
	__device__ bool map(int *testdata, T *object, void *global_data, const void *offset)
	{
		//atomicAdd(testdata, 1);
		float *mat1 = (float *)global_data;		
		float *mat2 = (float *)global_data+DIM*DIM;
		pos_t p = *(const pos_t *)offset;
		float * a = mat1 + p.x*DIM;
		float * b = mat2;
		float value = 0;

		for(int j = 0; j<DIM; j++)
		{
			value += a[j]*b[j*DIM+p.y];
		}

		unsigned int key = p.x*DIM+p.y;
		object->insert_or_update_value(testdata, &key, sizeof(unsigned int), &value, sizeof(float));
		return true;
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	__device__ void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
	{
	}

	__device__ int compare(void *&key1, unsigned short &key1_size, void *&key2, unsigned short &key2_size, void *&value1, unsigned short value1_size,  void *&value2, unsigned short value2_size)
	{
		unsigned int a, b;
		copyVal(&a, key1, sizeof(unsigned int));
		copyVal(&b, key2, sizeof(unsigned int));

		if(a>b)
			return 1;
		else if(a<b)
			return -1;
		else
			return 0;
	}
}

inline __device__ bool equal(void *key1, const unsigned int size1, void *key2, const unsigned int size2)
{
	if(size1!=size2)
		return false;

	char *k1 = (char *)key1;
	char *k2 = (char *)key2;

	for(int i = 0; i < size1; i++)
	{
		if(k1[i]!=k2[i])
			return false;
	}

	return true;
}

