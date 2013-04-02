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
	__device__ inline bool map(int *testdata, T *object, void *global_data, void *offset)
	{
		const float pass = *(float *)global_data;
		if(pass==0)
		{
		const pos_t pos = *(pos_t *)offset;
		float *data = (float *)global_data;
		const unsigned int n = (unsigned int)(*(data+1));
		const unsigned int m = (unsigned int)(*(data+2));
		data ++;
		data ++;

		//data points to the real data
		const unsigned int i = pos.x;
		const unsigned int j = pos.y;

		float value = 0;

		for(int row = 0; row < n; row++)
		{
			value+=data[m*row+i]*data[m*row+j];
		}

		unsigned int key = i*m+j;
		return object->insert_or_update_value(testdata, &key, sizeof(key), &value, sizeof(float));
		}
		else
			return true;
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	__device__ void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
	{
		unsigned int num_points1 = ((float *)value1)[3];
		unsigned int num_points2 = ((float *)value2)[3];
		unsigned int total_points = num_points1+num_points2;
		float temp[4];
		temp[0] = (((float *)value1)[0]*num_points1 + ((float *)value2)[0]*num_points2)/total_points;
		temp[1] = (((float *)value1)[1]*num_points1 + ((float *)value2)[1]*num_points2)/total_points;
		temp[2] = (((float *)value1)[2]*num_points1 + ((float *)value2)[2]*num_points2)/total_points;
		temp[3] = total_points;
	
		copyVal(value1, temp, sizeof(float)*4);
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

