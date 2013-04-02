//__device__ unsigned int hash(void *key, unsigned short size)
//{
//	unsigned int hs = 5381;
//	char *str = (char *)key;
//
//	for(int i = 0; i<size; i++)
//	{
//		hs = ((hs << 5) + hs) + ((int)str[i]); /* hash * 33 + c */
//	}
//	return hs;
//}

__device__ unsigned int hash(void *key, unsigned short size)
{
	   	unsigned int m;
		copyVal(&m, key, size);
	//	return *(unsigned int *)key;
		return m;
}
