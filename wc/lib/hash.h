__device__ unsigned short hash(void *key, unsigned short size)
{
	unsigned short hs = 5381;
	char *str = (char *)key;

	for(int i = 0; i<size; i++)
	{
		hs = ((hs << 5) + hs) + ((int)str[i]); /* hash * 33 + c */
	}
	return hs;
}
