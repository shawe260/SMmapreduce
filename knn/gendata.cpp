#include <iostream>
using namespace std;
#include <fstream>
#include <stdlib.h>
#define DIM 3
#define BSIZE 10000000 
#define GRIDSZ 1000

int main()
{
	ofstream output;
	char filename[] = "data";
	output.open(filename);
	int padding1 = DIM;
	int padding2 = BSIZE;
	output.write((char *)&padding1, sizeof(int));
	output.write((char *)&padding2, sizeof(int));
	double tmp;

    srand(2006);

    for(int i = 0; i<BSIZE*DIM; i++)
	{
	    tmp = rand()%GRIDSZ;	
		output.write((char *)&tmp, sizeof(double));
	}
	output.close();
	return 0;
}
