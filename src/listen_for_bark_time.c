/*
Copyright (c) 2014, Michael (Misko) Dzamba.
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


//#define COMPARE
#define CPU
//#define GPU


//#include <rfftw.h> 
#include <curl/curl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <semaphore.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>

#include "model.h"

#ifdef GPU
#include "mailbox.h"
#include "gpu_fft.h"
#endif

#ifdef CPU
#include <fftw3.h>
#endif

#define CAMERA_USB_MIC_DEVICE "plughw:1,0"

#define BARK_THRESHOLD 2.75

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


int mb;


char * json_buffer;
char device_id[1024];

sem_t s_ready; //means this many have been read and ready
sem_t s_done; //means N/2 have been processed
sem_t s_exit; //means N/2 have been processed
sem_t s_upload; //means N/2 have been processed

void short_to_double(double * real , signed short* shrt, int size) {
	int i;
	for(i = 0; i < size; ++i) {
		real[i] = shrt[i]; // 32768.0;
	}
}

int num_filters;

int buffer_frames = 2048*32;
//int buffer_frames = 2048*16;

#define NUM_BUFFERS 4
signed short ** raw_buffer_in;
double **buffer_in, **cpu_buffer_out, **gpu_buffer_out;
snd_pcm_t *capture_handle;
snd_pcm_hw_params_t *hw_params;
snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
fftw_plan p;

#define NUM_BARKS  4
#define BARK_BANK_SIZE	8 //256
unsigned int rate = 8000; //44100; //22050; //44100;
const int window_size= 256;
const int window_shift = 109;
int sample_frames = 2000; //0.25*fs
#define NON_OVERLAP (window_size-window_shift)
//#define WINDOWS (1+(sample_frames-window_size)/window_shift)
#define WINDOWS ((sample_frames-window_size)/window_shift)
#define WINDOW_AVG 4

int barks_total;
double * barks;
double median=0.0;
double * hanning_window;

double * bark_bank;
unsigned time_barks[2];

int saving_bark_bank,sending_bark_bank;
int stored_barks=0;
double barks_sum;

int exit_now=0;

int uploads=0;


float * filters=NULL;
float * biases=NULL;

double
hanning (int i, int nn) {
  return 0.54-0.46*cos(2*M_PI*(double)i/(double)(nn-1));
  //return ( 0.5 * (1.0 - cos (2.0*M_PI*(double)i/(double)(nn-1))) );
}

void normalize(double * d, int n, double m ) {
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n; i++) {
        mean+=d[i];
    }
    mean=mean/n;
    for(i=0; i<n; i++) {
    	sum_deviation+=(d[i]-mean)*(d[i]-mean);
    }
    double stddev=sqrt(sum_deviation/n);          
    for (i=0; i<n; i++) {
	d[i]-=mean;
	d[i]/=stddev;
	d[i]*=m;
    }
}

double stddev(double * d, int n) {
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n; i++) {
        mean+=d[i];
    }
    mean=mean/n;
    for(i=0; i<n; i++) {
    	sum_deviation+=(d[i]-mean)*(d[i]-mean);
    }
    return sqrt(sum_deviation/n);           
}


int cmp(const void *x, const void *y) {
  double xx = *(double*)x, yy = *(double*)y;
  if (xx < yy) return -1;
  if (xx > yy) return  1;
  return 0;
}

int cmp_p(const void *x, const void *y) {
  //fprintf(stderr,"%p %p\n",x,y);
  //fprintf(stderr,"\t%p %p\n",*((double **)x),*((double **)y));
  double xx = **((double**)x), yy = **((double**)y);
  if (xx < yy) return -1;
  if (xx > yy) return  1;
  return 0;
}


float * read_floats(char * fn ) {
	FILE * fptr = fopen(fn,"rb");
	if (fptr==NULL) {
		fprintf(stderr,"Failed to open file %s\n",fn);
		exit(1);
	}
	fseek(fptr, 0, SEEK_END);
	long size = ftell(fptr);
	fseek(fptr, 0, SEEK_SET);

	float *f = (float *)malloc(sizeof(float)*size);
	if(f==NULL) {
		fprintf(stderr,"FAILED TO MALLOC FOR BIN FILE!\n");
		exit(1);
	}


	size/=sizeof(float);
	//assert(size==(2048*4) || size==4);//hard coded filter sizese
	if(fread(f, sizeof(float), size, fptr)!=size) {
		fprintf(stderr,"FAILED TO READ BIN FILE!\n");
		exit(1);
	}

	fclose(fptr);

	return f;
}

unsigned microseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec*1000000 + ts.tv_nsec/1000;
}

void init_barks() {
	barks=(double*)malloc(sizeof(double)*NUM_BARKS);
	if (barks==NULL) {
		fprintf(stderr, "Failed to malloc for barks buffers\n");
		exit(1);
	}
	memset(barks,0,sizeof(double)*NUM_BARKS);

	bark_bank=(double*)malloc(sizeof(double)*BARK_BANK_SIZE*2);
	if (bark_bank==NULL) {
		fprintf(stderr, "Failed to malloc for store barks\n");
		exit(1);
	}
	memset(bark_bank,0,sizeof(double)*NUM_BARKS);
	
	saving_bark_bank=0;
	sending_bark_bank=0;
	stored_barks=0;

	json_buffer=malloc(sizeof(char)*5*BARK_BANK_SIZE+256);
	if (json_buffer==NULL) {
		fprintf(stderr,"FAILED TO ALLOCATE JSON BUFFER\n");
		exit(1);
	}
	
	time_barks[saving_bark_bank]=time(NULL); //microseconds();

	barks_total=0;
	barks_sum=0;
}


void add_bark_sum(double s) {
	if (stored_barks==BARK_BANK_SIZE) {
		//switch banks!
		//try to grab the mutex? -- assume already sent!
		if (saving_bark_bank!=sending_bark_bank) {
			return ; //not done sending previous!
		}
		stored_barks=0;
		saving_bark_bank=1-saving_bark_bank; //use the other bank
		time_barks[saving_bark_bank]=time(NULL); //microseconds();
		sem_post(&s_upload);
	}
	bark_bank[BARK_BANK_SIZE*saving_bark_bank+(stored_barks++)]=s;
	/*if (stored_barks%10==0) {
		fprintf(stderr,"STORED %d bark sin bank %d\n",stored_barks,saving_bark_bank);
	}*/
}



double sum_of_barks=0.0;
void add_bark(double b) {
	sum_of_barks-=barks[barks_total%NUM_BARKS];
	sum_of_barks+=b;
	barks[(barks_total++)%NUM_BARKS]=b;
}

double sum_barks() {
	return sum_of_barks; 
	/*
	double s=0.0;
	int i;
	for (i=0; i<NUM_BARKS; i++) {
		s+=barks[i];
	}
	fprintf(stderr,"DIFF IS %f %f %f\n",s,sum_of_barks,s-sum_of_barks);
	return s;*/
}

#ifdef GPU
struct GPU_FFT *gpu_fft;

int init_gpu() {
	//fprintf(stdout,"GPU init\n");
	int log2_N=11;
	//int jobs=1;
	//int loops=1;
	//int N = 1<<log2_N; //2048;
	mb = mbox_open();

    	//int ret = gpu_fft_prepare(mb, log2_N, GPU_FFT_REV, NUM_BUFFERS/2, &gpu_fft); // call once
    	int ret = gpu_fft_prepare(mb, log2_N, GPU_FFT_FWD, NUM_BUFFERS/2, &gpu_fft); // call once

    	switch(ret) {
        	case -1: printf("Unable to enable V3D. Please check your firmware is up to date.\n"); return -1;
        	case -2: printf("log2_N=%d not supported.  Try between 8 and 17.\n", log2_N);         return -1;
        	case -3: printf("Out of memory.  Try a smaller batch or increase GPU memory.\n");     return -1;
    	}	
	
	return 0;
}


void close_gpu() {
	mbox_close(mb);
}
#endif

int free_buffers() {
	free(buffer_in[0]);
	free(buffer_in);
	free(raw_buffer_in[0]);
	free(raw_buffer_in);
	return 0;
}

int init_buffers() {
	//fprintf(stdout,"Buffer init\n");
	
	double ** p = (double **)malloc(sizeof(double*)*NUM_BUFFERS);
	if (p==NULL) {
		fprintf(stdout,"Failed to malloc buffer pointers\n");
		exit(1);
	}
	buffer_in=p;

	p = (double **)malloc(sizeof(double*)*WINDOWS*NUM_BUFFERS);
	if (p==NULL) {
		fprintf(stdout,"Failed to malloc buffer pointers\n");
		exit(1);
	}
	cpu_buffer_out=p;

	p = (double **)malloc(sizeof(double*)*WINDOWS*NUM_BUFFERS);
	if (p==NULL) {
		fprintf(stdout,"Failed to malloc buffer pointers\n");
		exit(1);
	}
	gpu_buffer_out=p;
	
	double * b = (double *)malloc(sizeof(double)*buffer_frames*NUM_BUFFERS);
	if (b==NULL) {
		fprintf(stderr, "Failed to malloc buffers\n");
		exit(1);
	}
	memset(b,0,sizeof(double)*buffer_frames*NUM_BUFFERS);

	double * w = (double *)malloc(sizeof(double)*NUM_BUFFERS*WINDOWS*window_size);
	if (w==NULL) {
		fprintf(stderr, "Failed to malloc buffers\n");
		exit(1);
	}
	memset(w,0,sizeof(double)*NUM_BUFFERS*WINDOWS*window_size);

	double * wg = (double *)malloc(sizeof(double)*NUM_BUFFERS*WINDOWS*window_size);
	if (wg==NULL) {
		fprintf(stderr, "Failed to malloc buffers\n");
		exit(1);
	}
	memset(wg,0,sizeof(double)*NUM_BUFFERS*WINDOWS*window_size);

	int i;
	for (i=0; i<NUM_BUFFERS; i++) {
		buffer_in[i]=b+buffer_frames*i;	
		int j;
		for (j=0; j<WINDOWS; j++) {
			cpu_buffer_out[i*WINDOWS+j]=(w+i*WINDOWS*window_size)+j*window_size;//  buffer_frames*(i+NUM_BUFFERS);
			gpu_buffer_out[i*WINDOWS+j]=(wg+i*WINDOWS*window_size)+j*window_size;//   buffer_frames*(i+2*NUM_BUFFERS);
		}
	}


	//allocate and populate hanning window
	hanning_window=(double*)malloc(sizeof(double)*window_size);
	for (i=0; i<window_size; i++) {
		hanning_window[i]=hanning(i,window_size);
	}

	//allocate raw buffer in
	signed short ** p2 = (signed short**)malloc(sizeof(signed short*)*NUM_BUFFERS);
	if (p2==NULL) {
		fprintf(stdout,"Failed to malloc buffer pointers 2 \n");
		exit(1);
	}
	raw_buffer_in=p2;

	signed short * b2 = (signed short*)malloc(NUM_BUFFERS*buffer_frames * snd_pcm_format_width(format) / 8 * 2);
	if (b2==NULL) {
		fprintf(stderr, "Failed to malloc raw input buffer\n");
		exit(1);
	}
	for (i=0; i<NUM_BUFFERS; i++) {
		raw_buffer_in[i]=b2+buffer_frames*i;
	}

	memset(b2,0,NUM_BUFFERS*buffer_frames * snd_pcm_format_width(format) / 8 * 2);
	return 0;
}

#ifdef GPU
void fft_gpu(double * b_in, double * b_out,  int N) {
	struct GPU_FFT_COMPLEX *gpu_base = gpu_fft->in;
	//copy in the data
	int i;
	for (i=0; i<N; i++) {
		gpu_base[i].re=b_in[i];
		gpu_base[i].im=0;
	}
	//run the transform
        gpu_fft_execute(gpu_fft); // call one or many times
	//print out the values?
	for (i=0; i<N/2; i++) {
		b_out[i]=gpu_base[i].re;
		//fprintf(stdout,"%0.3f%c" , b_out[i], (i==N/2-1) ? '\n' : ',');
	}
	for (i=0; i<N/2; i++) {
		b_out[i+N/2]=gpu_base[i+N/2].im;
		//fprintf(stdout,"%0.3f%c" , b_out[i+N/2], (i==N/2-1) ? '\n' : ',');
	}
		
}

double fft_gpu_compare(double * d1, double *d2,  int N) {
	double x=0.0;
	int i;
	for (i=0; i<N; i++) {
		x+=fabs(fabs(d1[i]-d2[i]));
	}	
	return x;
}

#endif

void init_audio() {
  int err;
 
  if ((err = snd_pcm_open (&capture_handle, CAMERA_USB_MIC_DEVICE, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    fprintf (stderr, "cannot open audio device(%s)\n", 
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "audio interface opened\n");
		   
  if ((err = snd_pcm_hw_params_malloc (&hw_params)) < 0) {
    fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params allocated\n");
				 
  if ((err = snd_pcm_hw_params_any (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params initialized\n");
	
  if ((err = snd_pcm_hw_params_set_access (capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
    fprintf (stderr, "cannot set access type (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params access setted\n");
	
  if ((err = snd_pcm_hw_params_set_format (capture_handle, hw_params, format)) < 0) {
    fprintf (stderr, "cannot set sample format (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params format setted\n");
	
  if ((err = snd_pcm_hw_params_set_rate_near (capture_handle, hw_params, &rate, 0)) < 0) {
    fprintf (stderr, "cannot set sample rate (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params rate setted\n");
 
  //if ((err = snd_pcm_hw_params_set_channels (capture_handle, hw_params, 2)) < 0) {
  if ((err = snd_pcm_hw_params_set_channels (capture_handle, hw_params, 1)) < 0) {
    fprintf (stderr, "cannot set channel count (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params channels setted\n");
	
  if ((err = snd_pcm_hw_params (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot set parameters (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "hw_params setted\n");
	
  snd_pcm_hw_params_free (hw_params);
  fprintf(stdout, "hw_params freed\n");
	
  if ((err = snd_pcm_prepare (capture_handle)) < 0) {
    fprintf (stderr, "cannot prepare audio interface for use (%s)\n",
             snd_strerror (err));
    exit (1);
  }
  fprintf(stdout, "audio interface prepared\n");
}


void init_fftw3() {
  p = fftw_plan_r2r_1d(window_size, buffer_in[0], cpu_buffer_out[0] , FFTW_R2HC, FFTW_ESTIMATE);
}


void * read_audio(void * n) {
    //fprintf(stdout,"starting read_audio\n");
    int half=0;
    while (1) {
	    //wait for chunk to be done
	    //fprintf(stdout,"read_audio waiting for s_done\n");
	    sem_wait(&s_done);
	    //fprintf(stdout,"read_audio done waiting for s_done\n");
	    if (exit_now==1) {	
		sem_post(&s_ready);
		sem_post(&s_exit);
		return NULL;
	    }

	    int i;
	    for (i=0; i<NUM_BUFFERS/2; i++) {
		    int err;
		    if ((err = snd_pcm_readi (capture_handle, raw_buffer_in[i+half*NUM_BUFFERS/2], buffer_frames)) != buffer_frames) {
		      fprintf (stderr, "read from audio interface failed (%s)\n",
			       snd_strerror (err));
		      exit (1);
		    }
		    sem_post(&s_ready);
		    //fprintf(stderr,"READ %u\n",microseconds());
		    if (exit_now==1) {	
			sem_post(&s_exit);
			return NULL;
		    }
	    }

            //just read in a half
	    half=1-half;
    }
}




void process_window(double * d) {
	
	int k;
	double pws[WINDOW_AVG];
	memset(pws,0, sizeof(double)*WINDOW_AVG);
	for (k=2; k<10; k++) {
		int kk;
		for (kk=0; kk<WINDOW_AVG; kk++) {
			pws[kk]+=d[kk*window_size+k];
		}
	}
	double stddev_current = stddev(pws,WINDOW_AVG);
	int increasing=1;
	for (k=0; k<WINDOW_AVG-1; k++) {
		if (pws[k]>pws[k+1]) {
			increasing=0;
		}
		//fprintf(stderr, "%f," , pws[k]);
	}
	//fprintf(stderr, "%f\n" , pws[WINDOW_AVG-1]);
	

	int usable = 0;	
	if (stddev_current>100 && increasing==1) {
		usable=1;
	}	
	if (usable!=1) {
		return;
	}
	//fprintf(stderr,"STDDEV: %0.3f\n",stddev(d,window_size*WINDOW_AVG));

	double * tmp = (double*)malloc(sizeof(double)*window_size);
	memset(tmp, 0, sizeof(double)*window_size);
	double * tmp_v = (double*)malloc(sizeof(double)*window_size*WINDOW_AVG);
	memset(tmp_v, 0, sizeof(double)*window_size*WINDOW_AVG);
	//compute the mean over WINDOW_AVG windows/
	for (k=0; k<WINDOW_AVG; k++) {
		int h;
		for (h=0; h<window_size; h++) {
			tmp[h]=1.0/WINDOW_AVG * ( (k==0) ? d[h] : (tmp[h]+d[k*window_size+h])); 
		}
	}
	//subtract out the MEAN from these windows
	for (k=0; k<WINDOW_AVG; k++) {
		int h;
		for (h=0; h<window_size; h++) {
			tmp_v[k*window_size+h]=d[k*window_size+h]-tmp[h];
		}
	}

	//abs, log , and foldback
	for (k=0; k<WINDOW_AVG; k++) {
		int h;
		for (h=0; h<window_size/2; h++) {
			tmp_v[k]=log(abs(tmp_v[k*window_size+h])+abs(tmp_v[k*window_size + window_size-1-h])+1);
			tmp_v[k*window_size + window_size-1-h]=0; //clear the second half
		}
	}

	const double pr = 1-logit(tmp_v);
	//add_bark(d);
	add_bark(pr);
	//fprintf(stdout,"%f\n",d);
	time_t result = time(NULL);
	if (barks_total>NUM_BARKS && sum_barks()>BARK_THRESHOLD/8) {
		//printf("BARK detected at %f %f %s", pr, sum_barks(),  ctime(&result));
	}
	printf("at %f %f %s", pr, sum_barks(),  ctime(&result));
	add_bark_sum(sum_barks());
	free(tmp);
	free(tmp_v);

}

void * process_audio(void * n) {
	//fprintf(stdout,"starting process_audio\n");
	int half=0;
	while (1) {
		//move from raw in to input
		int i;
		for (i=0; i<NUM_BUFFERS/2; i++) {
			//fprintf(stderr,"PROCESS WAIT %u\n",microseconds());
			sem_wait(&s_ready);
			//fprintf(stderr,"PROCESS %u\n",microseconds());
			if (exit_now==1) {
				sem_post(&s_done);
				sem_post(&s_exit);
				return NULL;
			}
#ifdef GPU
			//COPY TO GPU		
			struct GPU_FFT_COMPLEX *gpu_base = gpu_fft->in+i*gpu_fft->step;
			
			int j;
	
			for (j=0; j<buffer_frames; j++) {
				raw_buffer_in[i+half*NUM_BUFFERS/2][j]=(j*j)%101; //DEBUG
				gpu_base[j].re=raw_buffer_in[i+half*NUM_BUFFERS/2][j];
				gpu_base[j].im=0;
			}
#endif
#ifdef CPU
			//COPY TO CPU
			short_to_double(buffer_in[i+half*NUM_BUFFERS/2],raw_buffer_in[i+half*NUM_BUFFERS/2],buffer_frames);
#endif			
				/*	//process this "bark"i
					char bf[128];
					sprintf(bf,"out%d\n",1111);
					FILE * fptr = fopen(bf,"wb");
					if (fptr==NULL) {
						fprintf(stderr,"FAILED TO OPEN FIL ETO\n");
						exit(1);
					}
					fwrite(buffer_in[i+half*NUM_BUFFERS/2],1,sizeof(double)*buffer_frames,fptr);
					fclose(fptr);
					exit(1);*/

			//normalize the signal
			//lets find the mean
			double s =0;
			int j;
			for (j=0; j<buffer_frames; j++) {
				s+=buffer_in[i+half*NUM_BUFFERS/2][j];
			}
			double mean = s/buffer_frames;
			double std = stddev(buffer_in[i+half*NUM_BUFFERS/2],buffer_frames);
			for (j=0; j<buffer_frames; j++) {
				buffer_in[i+half*NUM_BUFFERS/2][j]=1000*(buffer_in[i+half*NUM_BUFFERS/2][j]-mean)/std;
			}
			fprintf(stderr,"%lf %lf\n",mean,std);
			std = stddev(buffer_in[i+half*NUM_BUFFERS/2],buffer_frames);
			s=0;
			for (j=0; j<buffer_frames; j++) {
				s+=buffer_in[i+half*NUM_BUFFERS/2][j];
			}
			mean = s/buffer_frames;
			fprintf(stderr,"%lf %lf\n",mean,std);

	
		}

		unsigned t[4];

		//now we read in NUM_BUFFERS/2 chunks to GPU lets run this!
	    	//fprintf(stdout,"process_audio running gpu fft\n");
#ifdef GPU
		t[0]=microseconds();
		gpu_fft_execute(gpu_fft); 
		t[1]=microseconds();
		//copy out GPU vallues
		for (i=0; i<NUM_BUFFERS/2; i++) {
			struct GPU_FFT_COMPLEX *gpu_base = gpu_fft->out+i*gpu_fft->step;
			int j;
			for (j=0; j<buffer_frames/2; j++) {
				gpu_buffer_out[i+half*NUM_BUFFERS/2][j]=gpu_base[j].re;
			}
			for (j=0; j<buffer_frames/2; j++) {
				//gpu_buffer_out[i+half*NUM_BUFFERS/2][j+buffer_frames/2]=gpu_base[j+buffer_frames/2].im;
				gpu_buffer_out[i+half*NUM_BUFFERS/2][j+buffer_frames/2]=-gpu_base[j+buffer_frames/2].im; //i must have missed something somewhere, add -1 to fix..
			}
		}
#endif

#ifdef CPU

		for (i=0; i<NUM_BUFFERS/2; i++) {
			//lets do some peak finding
			double * candidate_barks = (double*)malloc(sizeof(double)*buffer_frames);
			memset(candidate_barks, 0, sizeof(double)*buffer_frames);

			double ** candidate_barks_p = (double**)malloc(sizeof(double*)*buffer_frames);
			memset(candidate_barks_p, 0, sizeof(double*)*buffer_frames);


			int peak_window_size = rate*0.25; //0.25ms - 2k at 8khz
			assert(peak_window_size==2000);
			double threshold_sum=0.0;
			int j;
			for (j=0; j<buffer_frames; j++) {
				threshold_sum+=abs(buffer_in[i+half*NUM_BUFFERS/2][j]);
			}
			double threshold=threshold_sum/buffer_frames; // mean(abs(z))
			

			//compute the thresholds for each window
			int s=0.0; double mx=0.0;
			for (j=0; j<buffer_frames; j++) {
				if (abs(buffer_in[i+half*NUM_BUFFERS/2][j])>threshold) {
					s+=1;
				}
				if (j>=peak_window_size) {
					if (abs(buffer_in[i+half*NUM_BUFFERS/2][j-peak_window_size])>threshold) {
						s-=1;
					}
					candidate_barks[j]=((double)s)/peak_window_size;
					if (candidate_barks[j]>mx) {
						mx=candidate_barks[j];
					}
				}
				candidate_barks_p[j]=candidate_barks+j;
			}

			//sort them and fill windows
			qsort(candidate_barks_p, buffer_frames, sizeof(double*), cmp_p);
			//fill the windows
			double * windows_buffer = (double*)malloc(sizeof(double)*WINDOWS*window_size);
			double * window_buffer = (double*)malloc(sizeof(double)*window_size);
			if (windows_buffer==NULL || window_buffer==NULL) {
				fprintf(stderr,"Failed to malloc window buffers\n");
				exit(1);
			}
			//fprintf(stderr,"mx %lf, f %lf, b %lf\n",mx, *candidate_barks_p[0], *candidate_barks_p[buffer_frames-1]);
			int yy=0;
			for (j=buffer_frames-1; j>0; j--) {
				size_t index = candidate_barks_p[j]-candidate_barks;	
				assert(*candidate_barks_p[j]==candidate_barks[index] || candidate_barks[index]<0);
				if (index>=peak_window_size) {
					//check this bark bounds
					if (candidate_barks[index-peak_window_size]<-1 || candidate_barks[index]<-1 || candidate_barks[index]<0.5) {
						continue;
					}
					fprintf(stderr,"PROCESSING CANDIDATE BARK %lf\n",candidate_barks[index]);
					//normalize per mean
					normalize(buffer_in[i+half*NUM_BUFFERS/2]+(index-peak_window_size),window_size+(WINDOWS-1)*window_shift,1000);
	
					//process this "bark"i
					/*char bf[128];
					sprintf(bf,"out%d\n",yy++);
					FILE * fptr = fopen(bf,"wb");
					if (fptr==NULL) {
						fprintf(stderr,"FAILED TO OPEN FIL ETO\n");
						exit(1);
					}
					fwrite(buffer_in[i+half*NUM_BUFFERS/2]+(index-peak_window_size),1,sizeof(double)*2000,fptr);
					fclose(fptr);
					if (yy==10) {
					exit(1);
					}*/
					int k;
					for (k=0; k<WINDOWS; k++ ){
						//copy and apply hamming
						memcpy(window_buffer, buffer_in[i+half*NUM_BUFFERS/2]+(index-peak_window_size)+k*window_shift, sizeof(double)*window_size);
						int h;
						//apply hanning before FFT
						for (h=0; h<window_size; h++) {
							window_buffer[h]=MAX(5.96e-8,MIN(65504,window_buffer[h]));
							window_buffer[h]*=hanning_window[h];
							//fprintf(stderr,"%e\n",hanning_window[h]);
						}
						//exit(1);
						/*for (h=0; h<window_size; h++) {
							window_buffer[h]=h*k;
						}*/
						//FFT
						//fftw_execute_r2r(p,buffer_in[i+half*NUM_BUFFERS/2]+j*window_shift,cpu_buffer_out[cpu_buffer_index]);
						fftw_execute_r2r(p,window_buffer,windows_buffer+window_size*k);
						//CPU and GPU differ on this value...
						windows_buffer[window_size*k+window_size/2]=0; //[cpu_buffer_index][window_size/2]=0; //fix sync issue between CPU and GPU
						//fold back and log
						for (h=0; h<window_size/2; h++) {
							windows_buffer[window_size*k+h]=abs( windows_buffer[window_size*k+h]) +abs(windows_buffer[window_size*(k+1)-1-h]);
							//windows_buffer[window_size*k+h]+=windows_buffer[window_size*(k+1)-1-h];
							windows_buffer[window_size*k+h]=log(abs(windows_buffer[window_size*k+h])+1);
							windows_buffer[window_size*(k+1)-1-h]=0;
						}
						for (h=0; h<14; h++) {
							windows_buffer[window_size*k+h]=0;
						}
						for (h=120; h<window_size/2; h++) {
							windows_buffer[window_size*k+h]=0;
						}
						/*for (h=0; h<window_size; h++) {
							fprintf(stderr,"%f %f\n",window_buffer[h],windows_buffer[window_size*k+h]);
						}
						fprintf(stderr,"%lf\n",log(10));
						exit(1);*/
					
					}
					//fill this bark
					for (k=0; k<peak_window_size; k++) {
						candidate_barks[index-1-k]=-2;
					}

					//analyze this bark
					double filter_vs[num_filters];
					for (k=0; k<num_filters; k++) {
						filter_vs[k]=0;
					}

	
					//VERY SPECIFIC AND HARD CODED :(
					//fprintf(stderr,"WINDOWS %d\n",WINDOWS);
					assert(WINDOWS==16);
					/*for (k=0; k<WINDOWS; k++) {
						int h;	
						for (h=0; h<128; h++) {
							//fprintf(stderr,"%0.2f,",windows_buffer[window_size*k+h]);
							//fprintf(stderr,"%0.4f,",filters[128*k+h]);
						}
						fprintf(stderr,"\n");
					}*/

					//float * one = read_floats("one");
					for (k=0; k<WINDOWS; k++) {
						int f;
						double s=0.0;
						for (f=0; f<num_filters; f++) {
							int h;
							for (h=0; h<window_size/2; h++) {
								filter_vs[f]+=windows_buffer[window_size*k+h]*filters[f*16*128+128*k+h];
								//filter_vs[f]+=one[window_size*k+h]*filters[f*16*128+128*k+h];
								//filter_vs[f]+=one[k*window_size/2+h]*filters[f*16*128+128*k+h];
								//fprintf(stderr,"%0.4f,",windows_buffer[window_size*k+h]*filters[f*16*128+128*k+h]);
								//fprintf(stderr,"%0.4f,",one[window_size*k + h]);
								//fprintf(stderr,"%0.4f,",one[k*window_size/2+h]);
								//s+=one[k+WINDOWS*h]*filters[f*16*128+128*k+h];
							}
							//fprintf(stderr,"\n");
						}
						//fprintf(stderr," | %e\n",s);
					}
					fprintf(stderr,"SCORE: ");
					//fprintf(stderr,"\n\n%e %e %e %e\n",filter_vs[0],filter_vs[1],filter_vs[2],filter_vs[3]);
					int f;
					for (f=0; f<num_filters; f++) {
						filter_vs[f]+=biases[f];
						if (filter_vs[f]<0) {
							filter_vs[f]=0.0;
						}
						fprintf(stderr,"%0.4f%c",filter_vs[f], f==num_filters-1 ? '\n' : ',');
					}
					const double pr = 1-logit(filter_vs);
					fprintf(stderr,"PR %e\n",pr);
					//exit(1);
					//exit(1);
				}
			}
			free(window_buffer);
			free(windows_buffer);
			free(candidate_barks);	
			free(candidate_barks_p);	
		}
		/*
	    	//fprintf(stdout,"process_audio running cpu fft\n");
		t[2]=microseconds();
		double * tmp = (double*)malloc(sizeof(double)*window_size);
		int * tmp_usable = (int*)malloc(sizeof(int)*WINDOWS);
		//for each raw_read of buffer input lets compute the ffts in the sliding window!
		for (i=0; i<NUM_BUFFERS/2; i++) {
			memset(tmp, 0, sizeof(double)*window_size);
			memset(tmp_usable, 0, sizeof(int)*WINDOWS);
			int j;
			for (j=0; j<WINDOWS; j++) {
							     //which buffer                    //which window
				const int cpu_buffer_index = (i+half*NUM_BUFFERS/2)* WINDOWS + j;
				memcpy(tmp, buffer_in[i+half*NUM_BUFFERS/2]+j*window_shift, sizeof(double)*window_size);
				int k;
				//apply hanning before FFT
				for (k=0; k<window_size; k++) {
					tmp[k]*=hanning_window[k];
				}

				//FFT
				//fftw_execute_r2r(p,buffer_in[i+half*NUM_BUFFERS/2]+j*window_shift,cpu_buffer_out[cpu_buffer_index]);
				fftw_execute_r2r(p,tmp,cpu_buffer_out[cpu_buffer_index]);

				//CPU and GPU differ on this value...
				cpu_buffer_out[cpu_buffer_index][window_size/2]=0; //fix sync issue between CPU and GPU
				//blank some out
				cpu_buffer_out[cpu_buffer_index][0]=0; //fix sync issue between CPU and GPU
				cpu_buffer_out[cpu_buffer_index][1]=0; //fix sync issue between CPU and GPU
				cpu_buffer_out[cpu_buffer_index][2]=0; //fix sync issue between CPU and GPU

				//abs and foldback
				//for (k=0; k<window_size/2; k++) {
				//	//fprintf(stderr,"%0.3e %0.3e %0.3e\n",tmp[k],tmp[k+window_size/2],tmp[window_size-1-k]);
				//	tmp[k]=abs(tmp[k])+abs(tmp[window_size-1-k]);
				//}

				//print window
				//for (k=0; k<window_size; k++) {
				//	fprintf(stderr,"%0.1f%c" , cpu_buffer_out[cpu_buffer_index][k], k==window_size-1 ? '\n' : ',');
				//}
				//exit(1);

			}
			//process the data
			
			//for each window lets compute abs and foldback
			for (j=0; j<WINDOWS-WINDOW_AVG; j++) {
				const int cpu_buffer_index = (i+half*NUM_BUFFERS/2)* WINDOWS + j;
				process_window(cpu_buffer_out[cpu_buffer_index]);
			}
		}	
		free(tmp);
		free(tmp_usable);
		t[3]=microseconds();*/
		//fprintf(stdout, "CPU %u\n",t[3]-t[2]);
#endif
		//uncomment to run CPU also


	
		if (exit_now==1) {
			sem_post(&s_done);
			sem_post(&s_exit);
			return NULL;
		}

#ifdef COMPARE
		fprintf(stdout, "GPU %u vs CPU %u\n",t[1]-t[0],t[3]-t[2]);
		
		for (i=0; i<buffer_frames; i++) {
			fprintf(stdout,"%0.3f%c" , cpu_buffer_out[0][i], (buffer_frames-1==i || i==buffer_frames/2-1) ? '\n' : ',');
		}

		for (i=0; i<buffer_frames; i++) {
			fprintf(stdout,"%0.3f%c" , gpu_buffer_out[0][i], (buffer_frames-1==i || i==buffer_frames/2-1) ? '\n' : ',');
		}

		//compare GPU and CPU on values
		for (i=0; i<NUM_BUFFERS/2; i++) {
			//lets find the difference between CPU and GPU computations
			double d =0.0;
			int j;
			for (j=0; j<buffer_frames; j++) {
				d+=fabs(gpu_buffer_out[i+half*NUM_BUFFERS/2][j]-cpu_buffer_out[i+half*NUM_BUFFERS/2][j]);	
			}	
			fprintf(stdout,"diff is %f\n", d);
		}
			exit(1);
#endif

/*		//prepare input
#ifdef GPU
		prepare_input(gpu_buffer_out+half*NUM_BUFFERS/2,10,buffer_frames);
#endif
#ifdef CPU
		prepare_input(cpu_buffer_out+half*NUM_BUFFERS/2,10,buffer_frames);
#endif


		for (i=0; i<NUM_BUFFERS/2; i++) {
#ifdef GPU
			const double d = logit(gpu_buffer_out[i+half*NUM_BUFFERS/2]);
#endif
#ifdef CPU
			const double d = logit(cpu_buffer_out[i+half*NUM_BUFFERS/2]);
#endif
			add_bark(d);
			//fprintf(stdout,"%f\n",sum_barks());
			add_bark_sum(sum_barks());
			if (barks_total>NUM_BARKS && sum_barks()<BARK_THRESHOLD) {
				time_t result = time(NULL);
				printf("BARK detected at %s\n", ctime(&result));
			}
		}
*/		
		sem_post(&s_done);
		if (exit_now==1) {
			sem_post(&s_exit);
			return NULL;
		}

		half=1-half;
	}
	
}

void cleanup() {
  //wait for threads to exit
#ifdef GPU
  gpu_fft_release(gpu_fft); // Videocore memory lost if not freed !
  close_gpu();
#endif
#ifdef CPU
  fftw_destroy_plan(p);
#endif
  snd_pcm_close (capture_handle);

  int i;
  for (i=0; i<NUM_BUFFERS; i++) {
	free(raw_buffer_in[i]);
	free(buffer_in[i]);
	free(cpu_buffer_out[i]);
	free(gpu_buffer_out[i]);
  }
  free_buffers();
  curl_global_cleanup();

}


void signal_callback_handler(int signum) {
   printf("Caught signal %d\n",signum);
   cleanup();
   exit(signum);
}


void update_mean(int bank, double scale) {

	//lets get the median instead...
	qsort(bark_bank+BARK_BANK_SIZE*sending_bark_bank, BARK_BANK_SIZE, sizeof(double), cmp);
	double median2=bark_bank[BARK_BANK_SIZE*sending_bark_bank+BARK_BANK_SIZE/32];
	median=median*scale+(1-scale)*median2;
	median=0;
	//fprintf(stderr,"MEAN IS %lf\n",mean);	
	//getting the mean
	/*
	int i;
	double s=0.0;
	for (i=0; i<BARK_BANK_SIZE; i++) {
		s+=bark_bank[BARK_BANK_SIZE*sending_bark_bank+i];
	}	
	s/=BARK_BANK_SIZE;
	
	mean=mean*scale+(1-scale)*s;*/
}

char * to_json() {

	//send all
	/*int x=0;
	x+=sprintf(json_buffer+x, "{ \"time-start\": %u, \"time-end\": %u , \"data\": [ " , time_barks[sending_bark_bank], microseconds());

	//lets subtract out the mean
	int i;	
	for (i=0; i<BARK_BANK_SIZE-1; i++) {
		double v = MAX(mean-bark_bank[BARK_BANK_SIZE*sending_bark_bank+i],0);
		x+=sprintf(json_buffer+x,"%0.1f,", v);
	}
	double v = MAX(mean-bark_bank[BARK_BANK_SIZE*sending_bark_bank+i],0);
	x+=sprintf(json_buffer+x,"%0.1f", v);
	x+=sprintf(json_buffer+x," ] }");
	fprintf(stderr,"JSON : %s\n",json_buffer);
	return json_buffer;*/

	//send sum
	//lets subtract out the mean
	double s=0.0;
	int i;
	for (i=0; i<BARK_BANK_SIZE; i++) {
		//double v = MAX(mean-bark_bank[BARK_BANK_SIZE*sending_bark_bank+i],0);
		//double v = mean-bark_bank[BARK_BANK_SIZE*sending_bark_bank+i];
		double v = median-bark_bank[BARK_BANK_SIZE*sending_bark_bank+i];
		s+=v;
	}
	s=abs(s);
	int x=0;
	x+=sprintf(json_buffer+x, "{ \"time-start\": %u, \"time-end\": %u , \"data\": %0.1f }" , time_barks[sending_bark_bank], time(NULL),s);
	//fprintf(stderr,"JSON : %s, %e, %e\n",json_buffer,s,median);
	return json_buffer;
}	

void * upload_barks(void * n ) {
	while (exit_now!=1) { 
		sem_wait(&s_upload); //wait for data to become available
		if (exit_now==1) {
			sem_post(&s_exit);
			return NULL;
		}
	  
		if (uploads<2) {
	  		update_mean(sending_bark_bank,0); //set the mean
		} else {
	  		update_mean(sending_bark_bank,0.9); //set the mean

			to_json();
			CURL *curl;
			CURLcode res;

			/* get a curl handle */
			curl = curl_easy_init();
			if(curl) {
				struct curl_slist *headers = NULL;
				headers = curl_slist_append(headers, "Accept: application/json");
				headers = curl_slist_append(headers, "Content-Type: application/json");
				headers = curl_slist_append(headers, "charsets: utf-8");

				curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers); 
				/* First set the URL that is about to receive our POST. This URL can
				   just as well be a https:// URL if that is what should receive the
				   data. */
				char url[1024];
				sprintf(url, "http://petbot.ca:1010/barks/%s", device_id);
				//fprintf(stderr,"URL is %s\n",url);
				curl_easy_setopt(curl, CURLOPT_URL, url);
				/* Now specify the POST data */
				curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_buffer);

				/* Perform the request, res will get the return code */
				res = curl_easy_perform(curl);
				/* Check for errors */
				if(res != CURLE_OK) {
					fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
				}
				/* always cleanup */
				curl_easy_cleanup(curl);
			}
		}
   
		sending_bark_bank=1-sending_bark_bank;
		uploads++;
	}
	sem_post(&s_exit);
	return NULL;
}


void read_device_id() {
	FILE * fptr= fopen("/proc/cpuinfo","r");
	if (fptr==NULL) {
		fprintf(stderr,"failed to read device id\n");
		exit(1);
	}
	char buffer[2048];
	size_t r = fread(buffer, 1, 2048, fptr);
	buffer[r-1]='\0';
	int i=1;
	while (!isalnum(buffer[r-i])) {
		buffer[r-i]='\0';
		i++;
	} 
	while (isalnum(buffer[r-i])) {
		i++;
	}
	i--;
	strcpy(device_id,buffer+r-i);
	return;
}

int main (int argc, char *argv[]) {
	signal(SIGINT, signal_callback_handler);
        signal(SIGTERM, signal_callback_handler);
  assert(buffer_frames%2==0);
  
  //fprintf(stdout,"reading model\n");

  if (argc!=3) {
      fprintf(stdout,"%s model_file filters\n",argv[0]);
      exit(1);
  }

  num_filters = atoi(argv[2]);
 
  char * model_fn=argv[1];

	filters=read_floats("filters");
	biases=read_floats("biases");
  read_device_id();

#ifdef GPU
  unlink(DEVICE_FILE_NAME); //dont really care about return value, just check mknod

  //dev_t d = makedev(major(100),minor(0));
  fprintf(stderr,"MAKE DEV\n");
  dev_t d = makedev(100,0);
  int r = mknod(DEVICE_FILE_NAME, S_IFCHR | S_IROTH | S_IRGRP | S_IRUSR | S_IWUSR, d);
  if (r<0) {
	fprintf(stderr,"Something went wrong with making file\n");
	exit(1);
  }
  fprintf(stderr,"MAKE DEV DONE\n");
#endif 
fprintf(stderr,"READMODEL\n");
  read_model(model_fn);
fprintf(stderr,"READMODEL\n");
  //fprintf(stdout,"starting inits\n");
  curl_global_init(CURL_GLOBAL_ALL);
fprintf(stderr,"READMODELz \n");
  init_barks();
fprintf(stderr,"READMODEL y \n");
  init_audio();
fprintf(stderr,"READMODEL x\n");
  init_buffers();
fprintf(stderr,"READMODEL 2\n");
#ifdef GPU
  init_gpu();
#endif
#ifdef CPU
  init_fftw3();
#endif

  //set up the semaphores
  sem_init(&s_ready, 0, 0); 
  sem_init(&s_done, 0, 2); 
  sem_init(&s_upload, 0, 0); 
  sem_init(&s_exit, 0, 0); 


  //compute the output frequencies
  /*double * freq = (double*)malloc(sizeof(double)*buffer_frames);
  if (freq==NULL) {
 	fprintf(stderr, "Failed to alloc freq array\n");
	exit(1);
  }
	
  int i;
  for (i = 0; i < buffer_frames; i++) {
	freq[i]=(((double)i)/buffer_frames)*rate;
	fprintf(stdout, "%f%c" , freq[i], (i==buffer_frames-1) ?  '\n' : ',');
  }*/

  //fprintf(stdout,"starting threads\n");

  pthread_t read_thread, process_thread, upload_thread;

  int iret1 = pthread_create( &read_thread, NULL, read_audio, NULL);
  if(iret1) {
    fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
    exit(1);
  }

  int iret2 = pthread_create( &process_thread, NULL, process_audio, NULL);
  if(iret2) {
    fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
    exit(1);
  }

  int iret3 = pthread_create( &upload_thread, NULL, upload_barks, NULL);
  if(iret3) {
    fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
    exit(1);
  }


  char line[2056];
  while (fgets(line, 2056, stdin)) {
	//fprintf(stdout,"still here!\n");
  }
  //fprintf(stdout,"Exiting!\n"); 
  exit_now=1;
  sem_post(&s_upload);
 
  sem_wait(&s_exit);
  sem_wait(&s_exit);
  sem_wait(&s_exit);

  cleanup();
	
  return 0;
  


}
