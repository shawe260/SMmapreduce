#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#ifndef KMEANS
#define KMEANS
#include "kmeans.h"
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
	__device__ bool map(int *testdata, T *object, void *global_data, void * offset)
	{
		float dim1 = ((float *)global_data)[*(unsigned int *)offset];
		float dim2 = ((float *)global_data)[*(unsigned int *)offset+1];
		float dim3 = ((float *)global_data)[*(unsigned int *)offset+2];
		unsigned int key = 0;
	
		float min_dist = 65536*65, dist; 
		//The first K points are the cluster centers
		for(int i = 0; i<K; i++)
		{
			dist = 0;
			float cluster_dim1 = ((float *)global_data)[DIM*i];
			float cluster_dim2 = ((float *)global_data)[DIM*i+1];
			float cluster_dim3 = ((float *)global_data)[DIM*i+2];
			
			dist = (cluster_dim1-dim1)*(cluster_dim1-dim1)+(cluster_dim2-dim2)*(cluster_dim2-dim2)+(cluster_dim3-dim3)*(cluster_dim3-dim3);
			dist = sqrt(dist);
			if(dist < min_dist)
			{
				min_dist = dist;
				key = i;
			}
		}
	
		float value[5];
		value[0] = dim1;
		value[1] = dim2;
		value[2] = dim3;
		value[3] = 1; 	//The last element of value records the number of one point, i.e., 1
		value[4] = min_dist;
	
		//you can choose which reduction object to use
		return object->insert_or_update_value(testdata, &key, sizeof(key), value, sizeof(float)*5);
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	__device__ void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
	{	
		unsigned int num_points1 = ((float *)value1)[3];
		unsigned int num_points2 = ((float *)value2)[3];
		float dist1 = ((float *)value1)[4];
		float dist2 = ((float *)value2)[4];
		unsigned int total_points = num_points1+num_points2;
		float temp[5];
		temp[0] = ((float *)value1)[0] + ((float *)value2)[0];
		temp[1] = ((float *)value1)[1] + ((float *)value2)[1];
		temp[2] = ((float *)value1)[2] + ((float *)value2)[2];
		temp[3] = total_points;
		temp[4] = dist1+dist2;
	
		copyVal(value1, temp, sizeof(float)*5);
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
}


