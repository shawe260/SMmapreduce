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

__device__ bool eq(char *str1, char *str2)
{
	int i = 0;
	char tmp1 = str1[0];
	char tmp2 = str2[0];
	while(tmp1==tmp2)
	{
		if(tmp1=='\0')
			return true;
		i++;
		tmp1 = str1[i];
		tmp2 = str2[i];
	}
	return false;
}

__device__ int get_color_d(char *c)
{
	if(eq(c, "Red"))
		return 0;
	else if(eq(c, "Yellow"))
		return 1;
	else if(eq(c, "White"))
		return 2;
	else return -1;
}

__device__ int get_type_d(char *t)
{
	if(eq(t, "Sports"))
		return 0;
	else if(eq(t, "SUV"))
		return 1;
	else if(eq(t, "Luxury"))
		return 2;
	else return -1;
}

__device__ int get_origin_d(char *o)
{
	if(eq(o, "USA"))
		return 0;
	else if(eq(o, "JP"))
		return 1;
	else if(eq(o, "GM"))
		return 2;
	else return -1;
}

__device__ int get_transmission_d(char *tr)
{
	if(eq(tr, "Manual"))
		return 0;
	else if(eq(tr, "Auto"))
		return 1;
	else if(eq(tr, "Combine"))
		return 2;
	else return -1;
}

__device__ int get_stolen_d(char *s)
{
	if(eq(s, "Yes"))
		return 0;
	else if(eq(s, "No"))
		return 1;
	else return -1;
}

__device__ int get_age_d(char *a)
{
	if(eq(a, "1"))
		return 1;
	else if(eq(a, "2"))
		return 2;
	else if(eq(a, "3"))
		return 3;
	else if(eq(a, "4"))
		return 4;
	else if(eq(a, "5"))
		return 5;
	else return -1;
}

typedef struct Reduction_Object RO;

namespace Map_Reduce
{
	//Define the way of emitting key-value pairs
	template <class T1, class T2>
	__device__ bool map(int *testdata, T1 *object_s, T2 *object_g, void *global_data, void * offset)
	{
		int c_i = ((char *)global_data)[0];
		int t_i = ((char *)global_data)[1];
		int o_i = ((char *)global_data)[2];
		int tr_i = ((char *)global_data)[3];
		int s_i = ((char *)global_data)[4];

		unsigned int ofst = *(unsigned int *)offset;
		char *c, *t, *o, *tr, *s, *a;
		int count = 0;
		char *p = (char *)global_data + ofst;
		c = p;
		for(int i = 0; count<6; i++)
		{
			if(((char *)global_data)[ofst+i]=='\0')
			{
				count++;
				if(count==1)
					t = p + i + 1;
				else if(count==2)
					o = p + i + 1;
				else if(count==3)
					tr = p + i + 1;
				else if(count==4)
					s = p + i + 1;
				else if(count==5)
					a = p + i + 1;
			}
		}

		int c_d = get_color_d(c);
		int t_d = get_type_d(t);
		int o_d = get_origin_d(o);
		int tr_d = get_transmission_d(tr);
		int s_d = get_stolen_d(s);
		int age = get_age_d(a);
		age--;

		int keys[6];
		for(int i = 0; i<6; i++)
			keys[i] = -1;

		keys[0] = age+25;

		if(c_d==c_i)
			keys[1] = age*5;
		if(t_d==t_i)
			keys[2] = age*5+1;
		if(o_d==o_i)
			keys[3] = age*5+2;
		if(tr_d==tr_i)
			keys[4] = age*5+3;
		if(s_d==s_i)
			keys[5] = age*5+4;

		//atomicAdd(testdata, *a);
		/*insert the key-val pairs*/
		int j = 0; //records the index
		int val = 1;
		bool result;
		for(; j<6; j++)
		{
			if(keys[j]!=-1)	
			{
				result = object_s->insert_or_update_value(testdata, &keys[j], sizeof(int), &val, sizeof(int));	
			}
			
			//the object is full
			if(!result) 			
				break;
		}

		if(result)
			return true;
		else
			return false;

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

    __device__ bool equal(void *key1, const unsigned int size1, void *key2, const unsigned int size2)
    {
    	int k1, k2;
    	copyVal(&k1, key1, sizeof(int));
    	copyVal(&k2, key2, sizeof(int));
    	return k1==k2;
    }
}



