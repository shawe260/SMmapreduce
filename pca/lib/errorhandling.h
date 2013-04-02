#  define CUT_CHECK_ERROR(errorMessage) do {				 \
    cudaError_t err = cudaGetLastError();				    \
    if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
    }									\
    err = cudaThreadSynchronize();					   \
    if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
    } } while (0)

#  define CE(call) do {                                \
	call;CUT_CHECK_ERROR("------- Error ------\n"); \
     } while (0)

