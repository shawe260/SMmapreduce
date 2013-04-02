#define int32 int
#define NUM_BUCKETS 600 //The number of buckets in a shared memory reduction object
#define MAX_POOL_IN_OBJECT 2000	//The max shared memory that can be used in each object, 200 ints 
#define NUM_BUCKETS_G 16384 //The number of buckets in a global memory reduction object
#define MAX_POOL_IN_OBJECT_G 4*1024*1024 //The max global memory that can be used in each global reduction object, 1M ints
#define ALIGN_SIZE 4

