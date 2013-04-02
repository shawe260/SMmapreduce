#define NUM_GROUPS 1//should be devidable by NUM_BLOCKS
#define NUM_BLOCKS 30  
#define NUM_THREADS 256 
#define USE_SHARED
#define WARP_SIZE 32
#define MERGE_WHEN_FULL
