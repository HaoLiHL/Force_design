#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(int argc, char *argv[])
{
	int buffersize = 512;
	char str[buffersize];
	// int num_atom = 12;

	// read energy, position and force file in the command line
	FILE *fp1, *fp2;

	// count number of lines in the energy file
	int line_energy=0;
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
		line_energy++;

	}
	fclose(fp1);

	// printf("%d\n", line_energy);

	int i, ii;
	char str1[buffersize];
	double energy;
	double index;
	char element_name[3];
	fp1 = fopen(argv[1], "r");
	fp2 = fopen(argv[2], "w+");

	double hartree2eV = 27.2114;
	double gradient2force = -1.0*27.2114/0.529177;
	
	for (i=0; i<line_energy; i++)
	{
		fscanf(fp1, "%s %lf", str1, &energy);
		fgets(str, buffersize, fp1);
		fprintf(fp2, "%.10lf\n", energy*hartree2eV);
	}
	fclose(fp1);
	fclose(fp2);
}
