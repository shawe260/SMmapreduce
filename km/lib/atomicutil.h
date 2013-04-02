//this function supports cas of unsigned int and float
//if compare equals *addr, then update the data in addr to val and return true; otherwise, return false and the data in addr remains unchanged
template <class T1, class T2, class T3>
__device__ bool CAS32(T1 *addr, T2 compare, T3 val)
{
	return *(unsigned int *)(&compare)==atomicCAS((unsigned int *)addr, *(unsigned int *)(&compare), *(unsigned int *)(&val));
}

template <class T1, class T2, class T3>
__device__ bool CAS64(T1 *addr, T2 compare, T3 val)
{
	return *(unsigned long long int*)(&compare)==atomicCAS((unsigned long long int*)addr, *(unsigned long long int*)(&compare), *(unsigned long long int*)(&val));
}

//Get the lock, when lockval = 0
__device__ bool getLock(int *lockVal)
{
	return atomicCAS(lockVal, 0, 1) == 0;
}

//release the lock, change lockVal to 0
__device__ void releaseLock(int *lockVal)
{
	atomicExch(lockVal, 0);
}
