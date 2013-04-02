void corcol(float* data, int n, int m)
{
	float eps = 0.005;
	//mean is the array containing the means for every column
	//stddev is the array containing the standard deviation for every column
	float x, *mean, *stddev;
	int i, j;

	/* Allocate storage for mean and std. dev. vectors */

	mean = (float *)malloc(m*sizeof(float));
	stddev = (float *)malloc(m*sizeof(float));
	/* Determine mean of column vectors of input data matrix */
	
	/*Use mapreduce to calculate the mean*/

	for (j = 0; j < m; j++)
	{
		mean[j] = 0.0;
		for (i = 0; i < n; i++)
		{
			mean[j] += data[m*i+j];
		}
		mean[j] /= (float)n;
	}

	printf("\nMeans of column vectors:\n");

	for (j = 0; j < m; j++)  
	{
		printf("%7.1f",mean[j]);  
	}   
	printf("\n");

	/* Determine standard deviations of column vectors of data matrix. */

	for (j = 0; j < m; j++)
	{
		stddev[j] = 0.0;
		for (i = 0; i < n; i++)
		{
			stddev[j] += (   ( data[i*m+j] - mean[j] ) *
				( data[i*m+j] - mean[j] )  );
		}
		stddev[j] /= (float)n;
		stddev[j] = sqrt(stddev[j]);
		/* The following in an inelegant but usual way to handle
		near-zero std. dev. values, which below would cause a zero-
		divide. */
		if (stddev[j] <= eps) stddev[j] = 1.0;
	}

	printf("\nStandard deviations of columns:\n");
	for (j = 0; j < m; j++) { printf("%7.1f", stddev[j]); }
	printf("\n");

	/* Center and reduce the column vectors. */

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			data[i*m+j] -= mean[j];
			x = sqrt((float)n);
			x *= stddev[j];
			data[i*m+j] /= x;
		}
	}
	
	free(mean);
	free(stddev);

	return;
}

