// This is a template for sequential, OpenMP and MPI programs.
// By filling in the code to calculate angles and incrementing the histograms
// this will work as a sequential program.
//
// In exercise 1, redesign the template for OpenMP: galaxy_openmp.c
//
// In exercise 2, redesign the template for MPI: galaxy_mpi.c

// Compilation on dione:
//    module load gcc               // do this once when you log in
//
// For OpemMP programs, compile with
//    gcc -O3 -fopenmp -o galaxy_openmp galaxy_openmp.c -lm
// and run with
//    srun -N 1 -c 40 ./galaxy_openmp RealGalaxies_100k_arcmin.dat SyntheticGalaxies_100k_arcmin.dat omega.out
//
//
// For MPI programs, compile with
//    mpicc -O3 -o galaxy_mpi_corrected galaxy_mpi_corrected.c -lm
//
// and run with e.g. 100 cores
//    srun -n 100 ./galaxy_mpi_corrected RealGalaxies_100k_arcmin.txt SyntheticGalaxies_100k_arcmin.txt omega.out




// Uncomment as necessary
#include <mpi.h>
//#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int MemoryAllocatedCPU = 0L;

int main(int argc, char* argv[])
    {
    int parseargs_readinput(int argc, char *argv[]);
    struct timeval _ttime;
    struct timezone _tzone;

    int np;
	int id;
	MPI_Status status[4];
	MPI_Init(NULL,NULL);

	//Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	//Get the rank of the processes
	MPI_Comm_rank(MPI_COMM_WORLD, &id);





    pif = acosf(-1.0f);

    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    // store right ascension and declination for real galaxies here
    // Note: indices run from 0 to 99999 = 100000-1: realrasc[0] -> realrasc[99999]
    // realrasc[100000] is out of bounds for allocated memory!
    real_rasc        = (float *)calloc(100000L, sizeof(float));
    real_decl        = (float *)calloc(100000L, sizeof(float));

    // store right ascension and declination for synthetic random galaxies here
    rand_rasc        = (float *)calloc(100000L, sizeof(float));
    rand_decl        = (float *)calloc(100000L, sizeof(float));

    MemoryAllocatedCPU += 10L*100000L*sizeof(float);

    // read input data from files given on the command line
    if ( parseargs_readinput(argc, argv) != 0 ) {printf("   Program stopped.\n");return(0);}
    printf("   Input data read, now calculating histograms\n");

    long int histogram_DD[360] = {0L};
    long int histogram_DR[360] = {0L};
    long int histogram_RR[360] = {0L};
    MemoryAllocatedCPU += 3L*360L*sizeof(long int);

//  Your code to calculate angles and filling the histograms
//  helpful hint: there are no angles above 90 degrees!
//  histogram[0] covers  0.00 <=  angle  <   0.25
//  histogram[1] covers  0.25 <=  angle  <   0.50
//  histogram[2] covers  0.50 <=  angle  <   0.75
//  histogram[3] covers  0.75 <=  angle  <   1.00
//  histogram[4] covers  1.00 <=  angle  <   1.25
//  and so on until     89.75 <=  angle  <= 90.0

    // here goes your code to calculate angles and to fill in
    // histogram_DD, histogram_DR and histogram_RR
    // use as input data the arrays real_rasc[], real_decl[], rand_rasc[], rand_decl[]

    //Send data to ranks
	MPI_Bcast(real_rasc,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(real_decl,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(rand_rasc,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(rand_decl,100000,MPI_FLOAT,0,MPI_COMM_WORLD);

	for ( int i = 0; i<360; ++i) //because I had some conversion problems so I ensure everything is at 0
	{
		histogram_DD[i]=0L;
		histogram_DR[i]=0L;
		histogram_RR[i]=0L;
	}


    //NOW WE COMPUTE THE ANGlE VALUES
    int n = 100000 / np;
	int debut = id * n;


	for ( int i = debut; i < debut+n; ++i )
	{
		for ( int j = 0; j < 100000; ++j )
		{
			float c = sinf(real_decl[i]) * sinf(real_decl[j]) + cosf(real_decl[i]) * cosf(real_decl[j]) * cosf(real_rasc[i]-real_rasc[j]);
			if ( c > 1.0f ) c = 1.0f; //in order to avoid conversion errors
			float angle = acosf(c);
			angle = angle * 180.0f/ pif ;
			histogram_DD[(int)(4.0f * angle)] += 1L;
		}
	}


// FOR DD VALUES
	for ( int i = debut; i < debut+n; ++i )
	{
		for ( int j = 0; j < 100000; ++j )
		{
			float c = sinf(rand_decl[i]) * sinf(rand_decl[j]) + cosf(rand_decl[i]) * cosf(rand_decl[j]) * cosf(rand_rasc[i]-rand_rasc[j]);
			if ( c > 1.0f ) c = 1.0f; //in order to avoid conversion errors
			float angle = acosf(c);
			angle = angle * 180.0f/ pif ;
			histogram_RR[(int)(4.0f * angle)] += 1L;
		}
	}


     for ( int i = debut; i < debut+n; ++i )
        {
            for ( int j = 0; j < 100000; ++j )
            {
                float c = sinf(real_decl[i]) * sinf(rand_decl[j]) + cosf(real_decl[i]) * cosf(rand_decl[j]) * cosf(real_rasc[i]-rand_rasc[j]);
                if ( c > 1.0f ) c = 1.0f; //in order to avoid conversion errors
                float angle = acosf(c);
                angle = angle * 180.0f/ pif ;
                histogram_DR[(int)(4.0f * angle)] += 1L;
            }
        }

        //We create the total HIST
	long int *histogram_DD_total, *histogram_RR_total, *histogram_DR_total;
	if ( id == 0 )
	{
		histogram_DD_total = (long int *)calloc(360L, sizeof(long int));
		histogram_DR_total = (long int *)calloc(360L, sizeof(long int));
		histogram_RR_total = (long int *)calloc(360L, sizeof(long int));
		MemoryAllocatedCPU += 3L*360L*sizeof(long int);
		for ( int i = 0; i<360; ++i) //same thing
		{
			histogram_DD_total[i]=0L;
			histogram_DR_total[i]=0L;
			histogram_RR_total[i]=0L;
		}
	}


	//NOW WE FINALLY REDUCE IN ORDER TO HAVE THE TOTAL VALUES
	MPI_Reduce(histogram_DD, histogram_DD_total, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(histogram_DR, histogram_DR_total, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(histogram_RR, histogram_RR_total, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);








    if(id==0){
            // check point: the sum of all historgram entries should be 10 000 000 000
        long int histsum = 0L;
        int      correct_value=1;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_DD_total[i];
        printf("   Histogram DD : sum = %ld\n",histsum);
        if ( histsum != 10000000000L ) correct_value = 0;

        histsum = 0L;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_DR_total[i];
        printf("   Histogram DR : sum = %ld\n",histsum);
        if ( histsum != 10000000000L ) correct_value = 0;

        histsum = 0L;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_RR_total[i];
        printf("   Histogram RR : sum = %ld\n",histsum);
        if ( histsum != 10000000000L ) correct_value = 0;

        if ( correct_value != 1 )
           {printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);}

        printf("   Omega values for the histograms:\n");
        float omega[360];
        for ( int i = 0; i < 360; ++i )
            if ( histogram_RR_total[i] != 0L )
               {
               omega[i] = (histogram_DD_total[i] - 2L*histogram_DR_total[i] + histogram_RR_total[i])/((float)(histogram_RR_total[i]));
               if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i*0.25, (i+1)*0.25, omega[i]);
               }

        FILE *out_file = fopen(argv[3],"w");
        if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
        else
           {
           for ( int i = 0; i < 360; ++i )
               if ( histogram_RR_total[i] != 0L )
                  fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
           fclose(out_file);
           printf("   Omega values written to file %s\n",argv[3]);
           }

    }





    free(real_rasc); free(real_decl);
    free(rand_rasc); free(rand_decl);
	MPI_Finalize();

    printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
    gettimeofday(&_ttime, &_tzone);
    double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    printf("   Wall clock run time    = %.1lf secs\n",time_end - time_start);

    return(0);
}



int parseargs_readinput(int argc, char *argv[])
    {
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;

    if ( argc != 4 )
       {
       printf("   Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
       return(1);
       }
    if ( argc == 4 )
       {
       printf("   Running galaxy_openmp %s %s %s\n",argv[1], argv[2], argv[3]);

       real_data_file = fopen(argv[1],"r");
       if ( real_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open real data file %s\n",argv[1]);
          return(1);
          }
       else
	  {
          fscanf(real_data_file,"%d",&Number_of_Galaxies);
          if ( Number_of_Galaxies != 100000 )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[1]);
             fclose(real_data_file);
             return(1);
             }
          for ( int i = 0; i < 100000; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[1]);
                 fclose(real_data_file);
	         return(1);
	         }
	      real_rasc[i] = rasc*arcmin2rad;
	      real_decl[i] = decl*arcmin2rad;
	      }
           fclose(real_data_file);
	   printf("   Successfully read 100000 lines from %s\n",argv[1]);
	   }

       rand_data_file = fopen(argv[2],"r");
       if ( rand_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open random data file %s\n",argv[2]);
          return(1);
          }
       else
	  {
          fscanf(rand_data_file,"%d",&Number_of_Galaxies);
          if ( Number_of_Galaxies != 100000 )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[2]);
             fclose(rand_data_file);
             return(1);
             }
          for ( int i = 0; i < 100000; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[2]);
                 fclose(rand_data_file);
	         return(1);
	         }
	      rand_rasc[i] = rasc*arcmin2rad;
	      rand_decl[i] = decl*arcmin2rad;
	      }
          fclose(rand_data_file);
	  printf("   Successfully read 100000 lines from %s\n",argv[2]);
	  }
       out_file = fopen(argv[3],"w");
       if ( out_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open output file %s\n",argv[3]);
          return(1);
          }
       else fclose(out_file);
       }

    return(0);
    }



