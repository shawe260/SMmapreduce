#define DIM 2048 

#ifndef DS_H
#define DS_H

struct pos_t{
	unsigned int x;
	unsigned int y;
};

struct global_data_t{
	float * A;
	float * B;
	int dim;
};
#endif
