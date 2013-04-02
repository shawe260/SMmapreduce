#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#ifndef KMEANS
#define KMEANS
#include "knn.h"
#endif

//#ifndef REDUCTIONOBJECT 
//#define REDUCTIONOBJECT
//#include "reductionobject_s.cu"
//#endif

#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "lib/reductionobject_s.cu"
#endif


typedef struct Reduction_Object RO;

namespace Map_Reduce
{
	//Define the way of emitting key-value pairs
	template <class T>
	__device__ bool map(int* testdata, T* object, void *global_data, void *offset)
	{
		float key = ((float *)global_data)[*(unsigned int *)offset];
		float dim1 = ((float *)global_data)[*(unsigned int *)offset+1];
		float dim2 = ((float *)global_data)[*(unsigned int *)offset+2];
		float dim3 = ((float *)global_data)[*(unsigned int *)offset+3];

	//	float cluster_dim1 = ((float *)global_data)[1];
	//	float cluster_dim2 = ((float *)global_data)[2];
	//	float cluster_dim3 = ((float *)global_data)[3];
	//
	//	float dist = (cluster_dim1-dim1)*(cluster_dim1-dim1)+(cluster_dim2-dim2)*(cluster_dim2-dim2)+(cluster_dim3-dim3)*(cluster_dim3-dim3);
		float dist = (100-dim1)*(100-dim1)+(100-dim2)*(100-dim2)+(100-dim3)*(100-dim3);
		//float dist = 0;
		dist = sqrt(dist);
	
		float value[1];
		value[0] = dist;
	
		return object->insert_or_update_value(testdata, &key, sizeof(key), value, sizeof(value));
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	__device__ void reduce(void *value1,  void *value2)
	{
	}

	__device__ int compare(void *&key1, unsigned short &size1, void *&key2, unsigned short &size2, void *&value1, void *&value2)
	{

		float a = *(float *)value1;	
		float b = *(float *)value2;	

		if(a>b)
			return 1;
		else if(a<b)
			return -1;
		else
			return 0;
	} 
}

__device__ bool equal(void *key1, const unsigned int size1, void *key2, const unsigned int size2)
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

