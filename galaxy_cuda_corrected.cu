// on dione, first load the cuda module
//    module load cuda
//
// compile your program with
//    nvcc -O3 -arch=sm_70 --ptxas-options=-v -o galaxy_cuda_corrected galaxy_cuda_corrected.cu -lm
//
// run your program with 
//    srun -p gpu -c 1 --mem=10G ./galaxy_cuda_corrected RealGalaxies_100k_arcmin.txt SyntheticGalaxies_100k_arcmin.txt omega.out



#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;//value of pi
long int MemoryAllocatedCPU = 0L;
	
	
__global__ void  fillHistogram(float * real_rasc, float * real_decl, float * rand_rasc, float * rand_decl, unsigned long long int * histogramDR, unsigned long long int * histogramDD, unsigned long long int * histogramRR) {

	float angleDR, angleRR, angleDD;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	
	if(i < 100000){
		int j;
		
		float pif = acosf(-1.0f);
		float tempDR,tempDD,tempRR = 0.0f;
		

	
		
	    	for(j = 0;j < 100000;j++){// we are doing exactly the same technique as exercise 2

	    		tempDR = sinf(real_decl[i]) * sinf(rand_decl[j]) + cosf(real_decl[i]) * cosf(rand_decl[j]) * cosf(real_rasc[i]-rand_rasc[j]);
				angleDR = tempDR >= 1 ? 0 :180.0f/pif*acosf(tempDR);
	    		tempDD = sinf(real_decl[i])*sinf(real_decl[j])+cosf(real_decl[i])*cosf(real_decl[j])*cosf(real_rasc[i]-real_rasc[j]);
				angleDD = tempDD >= 1 ? 0 :180.0f/pif*acosf(tempDD);
	    		tempRR = sinf(rand_decl[i])*sinf(rand_decl[j])+cosf(rand_decl[i])*cosf(rand_decl[j])*cosf(rand_rasc[i]-rand_rasc[j]);
				angleRR = tempRR >= 1 ? 0 :180.0f/pif*acosf(tempRR);
	    		
			
			atomicAdd(&histogramDR[(int)(angleDR*4.0f)],1L);//allow for 1 thread to write at a time
			atomicAdd(&histogramDD[(int)(angleDD*4.0f)],1L);
			atomicAdd(&histogramRR[(int)(angleRR*4.0f)],1L);
	    			    		
	    	}

		
	}
	
}




int main(int argc, char* argv[])
    {
    int parseargs_readinput(int argc, char *argv[]);
	double start, end;
    struct timeval _ttime;
    struct timezone _tzone;
	pif = acosf(-1.0);
	int getDevice(void);


    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
	
	// For your entertainment: some performance parameters of the GPU you are running your programs on!
   if ( getDevice() != 0 ) return(-1);

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

// input data is available in the arrays float real_rasc[], real_decl[], rand_rasc[], rand_decl[];
   // allocate memory on the GPU for input data and histograms
   // and initialize the data on GPU by copying the real and rand data to the GPU
   size_t data_size = 100000*sizeof(float);
   size_t histo_size = 360*sizeof(unsigned long long int);
   
   float * real_rasc_gpu; cudaMalloc(&real_rasc_gpu,data_size);//We are defining the spaces needed in the GPU for our computation elements
   float * real_decl_gpu; cudaMalloc(&real_decl_gpu,data_size);
   float * rand_rasc_gpu; cudaMalloc(&rand_rasc_gpu,data_size);
   float * rand_decl_gpu; cudaMalloc(&rand_decl_gpu,data_size);
   unsigned long long int * histogramDR_gpu; cudaMalloc(&histogramDR_gpu, histo_size);//I tried with only long but the kernel functions works only with unsigned long long int
   unsigned long long int * histogramDD_gpu; cudaMalloc(&histogramDD_gpu, histo_size);//
   unsigned long long int * histogramRR_gpu; cudaMalloc(&histogramRR_gpu, histo_size);
   
   
   cudaMemset(histogramDR_gpu, 0, histo_size);//put everything at 0
   cudaMemset(histogramRR_gpu, 0, histo_size);
   cudaMemset(histogramDD_gpu, 0, histo_size);
   
   
   cudaMemcpy(real_rasc_gpu, real_rasc, data_size, cudaMemcpyHostToDevice);//we are copying the datas from the CPU to the GPU
   cudaMemcpy(real_decl_gpu, real_decl, data_size, cudaMemcpyHostToDevice);
   cudaMemcpy(rand_rasc_gpu, rand_rasc, data_size, cudaMemcpyHostToDevice);
   cudaMemcpy(rand_decl_gpu, rand_decl, data_size, cudaMemcpyHostToDevice);
   
 
   
   

   // call the GPU kernel(s) that fill the three histograms
   int threadsInBlock = 256;
   int blocksInGrid = (100000 + threadsInBlock - 1) /threadsInBlock;
   start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;


   fillHistogram<<<blocksInGrid, threadsInBlock>>>(real_rasc_gpu, real_decl_gpu,rand_rasc_gpu,rand_decl_gpu,histogramDR_gpu,histogramDD_gpu, histogramRR_gpu);

   if(cudaDeviceSynchronize() != cudaSuccess) return(-1);
   
   

    
    cudaMemcpy(histogram_DD, histogramDD_gpu, histo_size, cudaMemcpyDeviceToHost);//we are taking back our solutions values
    cudaMemcpy(histogram_DR, histogramDR_gpu, histo_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram_RR, histogramRR_gpu, histo_size, cudaMemcpyDeviceToHost);



    // here goes your code to calculate angles and to fill in
    // histogram_DD, histogram_DR and histogram_RR
    // use as input data the arrays real_rasc[], real_decl[], rand_rasc[], rand_decl[]





    // check point: the sum of all historgram entries should be 10 000 000 000
    long int histsum = 0L;
    int      correct_value=1;
    for ( int i = 0; i < 360; ++i ) histsum += histogram_DD[i];
    printf("   Histogram DD : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

    histsum = 0L;
    for ( int i = 0; i < 360; ++i ) histsum += histogram_DR[i];
    printf("   Histogram DR : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

    histsum = 0L;
    for ( int i = 0; i < 360; ++i ) histsum += histogram_RR[i];
    printf("   Histogram RR : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

    if ( correct_value != 1 )
       {printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);}

    printf("   Omega values for the histograms:\n");
    float omega[360];
    for ( int i = 0; i < 360; ++i )
        if ( histogram_RR[i] != 0L )
           {
           omega[i] = (histogram_DD[i] - 2L*histogram_DR[i] + histogram_RR[i])/((float)(histogram_RR[i]));
           if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i*0.25, (i+1)*0.25, omega[i]);
           }

    FILE *out_file = fopen(argv[3],"w");
    if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
    else
       {
       for ( int i = 0; i < 360; ++i )
           if ( histogram_RR[i] != 0L )
              fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
       fclose(out_file);
       printf("   Omega values written to file %s\n",argv[3]);
       }


    free(real_rasc); free(real_decl);
    free(rand_rasc); free(rand_decl);
    cudaFree(real_rasc_gpu); cudaFree(real_decl_gpu);
    cudaFree(rand_rasc_gpu); cudaFree(rand_decl_gpu);
    cudaFree(histogramDD_gpu); cudaFree(histogramDR_gpu);
    cudaFree(histogramRR_gpu);

    printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
    gettimeofday(&_ttime, &_tzone);
    double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
	end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    printf("   Wall clock run time    = %.1lf secs\n",time_end - time_start);
	printf("   Kernel run time    = %.1lf secs\n",end - start);

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






int getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio    =    %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                  =    %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim                =    %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                =    %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

