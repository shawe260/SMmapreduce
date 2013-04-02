#define int32 int
#define NUM_BUCKETS 22//The number of buckets in a shared memory reduction object, make it power of 2, which is necessary for bitonic sorting
#define MAX_POOL_IN_OBJECT 700	//The max shared memory that can be used in each object, 200 ints 
#define NUM_BUCKETS_G 10000000 //The number of buckets in a global memory reduction object, make it power of 2, which is necessary for bitonic sorting
#define MAX_POOL_IN_OBJECT_G 64*1024*1024 //The max global memory that can be used in each global reduction object, 1M ints
#define ALIGN_SIZE 4
#define SORT_REMAIN 10 

