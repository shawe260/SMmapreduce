#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#define NUM 10000000

using namespace std;

/*There are four fields in the data set
 *
 * Color, Type, Origin, Age, Transmission, and Stolen * There are three kinds of colors: Red, Yellow, and White * There are three types: Sports, SUV, and Luxury
 * There are three origins: USA, JP, and GM
 * There are two possibility of stolen: Yes, No
 * */

int main()
{
	ofstream output;
	char filename[] = "dataset";
	output.open(filename);
	srand ( time(NULL) );

	for(int i = 0; i < NUM; i++)
	{
		/*generate color index*/	
		int c = rand()%3; 
		if(c==0)
			output.write("Red\t", 4);
		else if(c==1)
			output.write("Yellow\t", 7);
		else if(c==2)
			output.write("White\t", 6);

		/*generate type*/
		int t = rand()%3; 
		if(t==0)
			output.write("Sports\t", 7);
		else if(t==1)
			output.write("SUV\t", 4);
		else if(t==2)
			output.write("Luxury\t", 7);

		/*generate origin*/
		int o = rand()%3; 
		if(o==0)	
			output.write("USA\t", 4);
		else if(o==1)
			output.write("JP\t", 3);
		else if(o==2)
			output.write("GM\t", 3);

		
		/*generate transmission*/
		int tr = rand()%3;
		if(tr==0)
			output.write("Manual\t", 7);
		else if(tr==1)
			output.write("Auto\t", 5);
		else if(tr==2)
			output.write("Combine\t", 8);

		/*generate stolen*/
		int s = rand()%2;
		if(s==0)
			output.write("Yes\t", 4);
		else if(s==1)
			output.write("No\t", 3);

		/*generate age*/
		int a = rand()%5+1;
		char str[3];
		sprintf(str, "%d", a);
		output.write(str, 1);
		output.write("\n", 1);
	}

	output.close();
	return 0;
}
