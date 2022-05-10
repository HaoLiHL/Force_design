#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(int argc, char *argv[])
{
	int buffersize = 512;
	char str[buffersize];
	int num_atom = 12;

	FILE *fp1, *fp2;

	// read total lines in the document
	int total_line=0;
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
		total_line++;

	}
	fclose(fp1);
	// calculate how many configurations in the output file
	int num_configuration = (int) ceil(1.0*total_line/(ceil(1.0*num_atom/6)*4+2));

	int i, ii, iii, j;
	char str1[buffersize];
	double index;
	fp1 = fopen(argv[1], "r");
	fp2 = fopen(argv[2], "w+");

	double gradient2force = -1.0*627.5/0.529177;

	int line_full_six = num_atom/6;
	int reminder = num_atom - line_full_six*6;
	// printf("%d %d\n", line_full_six, reminder);

	double x_gradient[6], y_gradient[6], z_gradient[6];
	
	for (i=0; i<num_configuration; i++)
	{
		fgets(str, buffersize, fp1);

		for (ii=0; ii<line_full_six; ii++)
		{
			fgets(str, buffersize, fp1);
			fscanf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", &index, &x_gradient[0], &x_gradient[1], &x_gradient[2], &x_gradient[3], &x_gradient[4], &x_gradient[5]);
			fscanf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", &index, &y_gradient[0], &y_gradient[1], &y_gradient[2], &y_gradient[3], &y_gradient[5], &y_gradient[5]);
			fscanf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", &index, &z_gradient[0], &z_gradient[1], &z_gradient[2], &z_gradient[3], &z_gradient[4], &z_gradient[5]);

			for (iii=0; iii<6; iii++)
			{
				fprintf(fp2, "%.7lf %.7lf %.7lf\n", x_gradient[iii]*gradient2force, y_gradient[iii]*gradient2force, z_gradient[iii]*gradient2force);
			}
		}

		if (reminder>0)
		{
			fgets(str, buffersize, fp1);
			fscanf(fp1, "%lf ", &index);
			for (ii=0; ii<reminder; ii++)
			{
				fscanf(fp1, "%lf ", &x_gradient[ii]);
			}
			fscanf(fp1, "%lf ", &index);
			for (ii=0; ii<reminder; ii++)
			{
				fscanf(fp1, "%lf ", &y_gradient[ii]);
			}
			fscanf(fp1, "%lf ", &index);
			for (ii=0; ii<reminder; ii++)
			{
				fscanf(fp1, "%lf ", &z_gradient[ii]);
			}
			for (iii=0; iii<reminder; iii++)
			{
				fprintf(fp2, "%.7lf %.7lf %.7lf\n", x_gradient[iii]*gradient2force, y_gradient[iii]*gradient2force, z_gradient[iii]*gradient2force);
			}
		}
		
		fgets(str, buffersize, fp1);
	}
	fclose(fp1);;
	fclose(fp2);;
}
