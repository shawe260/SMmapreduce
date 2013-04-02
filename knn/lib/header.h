#define NUM_GROUPS 4 //should be devidable by NUM_BLOCKS
#define NUM_BLOCKS 30  
#define NUM_THREADS 256 //should be at least equal to number of buckets, which is necessary for sorting
#define USE_SHARED
/*The following two macros indicate the actions to be taken when the shared reduction object is full, either of them is used*/
//#define MERGE_WHEN_FULL
#define SORT_WHEN_FULL
