#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
using namespace std;

#define FILE_NUMBER 1
//#define ROW 34560
#define ROW 1048576 
#define COLUMN 64 
double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
};

int main(int argc, char **argv) {
    long seedval=1000;
	double c0 = rtclock();
    srand48((int)c0);

    char output_file[256];
    char index_file[256];
    
    ofstream output_fp;
 
    int i,j,k,l;

    strcpy(index_file,"data.index"); 
    
    int tid;
    int nitems;
	float item;
    char *str;
    char postfix[9];
     
    int low=0;
    int high=1;
    int fd;
    int offset;
    int size; 
    
    int temp;

    for (i=0;i<FILE_NUMBER;i++) {
   
    output_fp.open("input",ios::out);
	cout<<"OK?"<<endl;
    fd=i;
    offset=0; 

    for (j=0;j<ROW;j++) {
             
    for (k=0;k<COLUMN;k++){

	item=drand48()*180;
	output_fp<<item<<" "; //, sizeof(float));
              }
             
        }
    output_fp.close();
    }   
}

