#ifndef UTIL
#define UTIL
#include "lib/util.h"
#endif

#ifndef REDUCTIONOBJECTS 
#define REDUCTIONOBJECTS
#include "lib/reductionobject_s.cu"
#endif

#ifndef DS_h
#define DS_h
#include "DS.h"
#endif

__device__ int StrHash(char *str, int strLen)
{
        int hash = strLen;

        for(int i=0; i < strLen; i++)
        	hash = (hash<<4)^(hash>>28)^str[i];

        return hash;
}

typedef struct Reduction_Object RO;

namespace Map_Reduce
{
	//Define the way of emitting key-value pairs
	template <class T1, class T2>
	__device__ bool map(int *testdata, T1 *object_s, T2 *object_g, void *global_data, void * offset)
	{
		int pass = *(char *)global_data;
		unsigned int ofst = *(unsigned int *)offset;
		if(pass==0)
		{
			char *line = (char *)global_data + ofst;
			char *curr = line;
			while(*curr!='\0')
				curr++;
			unsigned int len = curr-line+1;
			int value = 1;//len;
		//	Key k;
		//	k.x = StrHash(line, len);
		//	k.y = ofst;
			
			return object_s->insert_or_update_value(testdata, line, len, &value, sizeof(unsigned int));
		}

		else
		{
			char *line = (char *)global_data+ofst;	
			char *curr = line;
			while(*curr!='\t')
				curr++;
			int len = curr-line;
			Key k;
			k.x = StrHash(line, len);
			k.y = ofst;
			int v = 1;
			return object_s->insert_or_update_value(testdata, &k, sizeof(k), &v, sizeof(unsigned int));
		}
	}
	
	//Define the operation between value1 and value2 and reduce the result to value1
	__device__ void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
	{	
		int sum = *(int *)value1 + *(int *)value2;
		copyVal(value1, &sum, sizeof(int));
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
//	Key k1, k2;
//	copyVal(&k1, key1, sizeof(Key));
//	copyVal(&k2, key2, sizeof(Key));
//	return k1.x==k2.x;
}

