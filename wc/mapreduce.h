#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif


#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "lib/reductionobject_s.cu"
#endif

typedef struct Reduction_Object RO;

namespace Map_Reduce
{
	//Define the way of emitting key-value pairs
	template <class T1, class T2>
	__device__ bool map(int *testdata, T1 *object_s, T2 *object_g, void *global_data, void *offset)
	{
		char *key = (char *)global_data + *(int *)offset; //The start of the word 
		char *p = key;

		int key_size = 1;
		while(*p!='\0')
		{
			key_size++;
			p++;	
		};

		//atomicAdd(testdata, key_size);
		
		int value = 1;
			
		return object_s->insert_or_update_value(testdata, key, key_size, &value, sizeof(value));
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	
	__device__ void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
	{
		int temp = *(int *)value1 + *(int *)value2;
		copyVal(value1, &temp, sizeof(int));
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



