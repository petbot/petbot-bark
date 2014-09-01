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

#include <fftw3.h>

//#include <rfftw.h> 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <semaphore.h>
#include <pthread.h>
#include <time.h>

#include "model.h"
#include "mailbox.h"
#include "gpu_fft.h"



#define CAMERA_USB_MIC_DEVICE "plughw:1,0"


#define BARK_THRESHOLD 2.5

	      
sem_t s_ready; //means this many have been read and ready
sem_t s_done; //means N/2 have been processed
sem_t s_exit; //means N/2 have been processed


void short_to_double(double * real , signed short* shrt, int size) {
	int i;
	for(i = 0; i < size; ++i) {
		real[i] = shrt[i]; // 32768.0;
	}
}


int buffer_frames = 2048;
unsigned int rate = 8000; //44100; //22050; //44100;
#define NUM_BUFFERS 20
signed short ** raw_buffer_in;
double **buffer_in, **cpu_buffer_out, **gpu_buffer_out;
snd_pcm_t *capture_handle;
snd_pcm_hw_params_t *hw_params;
snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
fftw_plan p;

#define NUM_BARKS  8
int barks_total;
double * barks;
double barks_sum;


int exit_now=0;

void init_barks() {
	barks=(double*)malloc(sizeof(double)*NUM_BARKS);
	if (barks==NULL) {
		fprintf(stderr, "Failed to malloc for barks buffers\n");
		exit(1);
	}
	memset(barks,0,sizeof(double)*NUM_BARKS);
	barks_total=0;
	barks_sum=0;
}

void add_bark(double b) {
	barks[(barks_total++)%NUM_BARKS]=b;
}




double sum_barks() {
	double s=0.0;
	int i;
	for (i=0; i<NUM_BARKS; i++) {
		s+=barks[i];
	}
	return s;
}




struct GPU_FFT *gpu_fft;

int init_gpu() {
	fprintf(stdout,"GPU init\n");
	int log2_N=11;
	//int jobs=1;
	//int loops=1;
	//int N = 1<<log2_N; //2048;
	int mb = mbox_open();


    	int ret = gpu_fft_prepare(mb, log2_N, GPU_FFT_REV, NUM_BUFFERS/2, &gpu_fft); // call once

    	switch(ret) {
        	case -1: printf("Unable to enable V3D. Please check your firmware is up to date.\n"); return -1;
        	case -2: printf("log2_N=%d not supported.  Try between 8 and 17.\n", log2_N);         return -1;
        	case -3: printf("Out of memory.  Try a smaller batch or increase GPU memory.\n");     return -1;
    	}	
	
	

	return 0;
}


int free_buffers() {
	free(buffer_in[0]);
	free(buffer_in);
	free(raw_buffer_in[0]);
	free(raw_buffer_in);
	return 0;
}

int init_buffers() {
	fprintf(stdout,"Buffer init\n");
	
	double ** p = (double **)malloc(sizeof(double*)*NUM_BUFFERS*3);
	if (p==NULL) {
		fprintf(stdout,"Failed to malloc buffer pointers\n");
		exit(1);
	}
	buffer_in=p;
	cpu_buffer_out=p+NUM_BUFFERS;
	gpu_buffer_out=p+NUM_BUFFERS*2;
	
	double * b = (double *)malloc(sizeof(double)*buffer_frames*NUM_BUFFERS*3);
	if (b==NULL) {
		fprintf(stderr, "Failed to malloc buffers\n");
		exit(1);
	}
	int i;
	for (i=0; i<NUM_BUFFERS; i++) {
		buffer_in[i]=b+buffer_frames*i;	
		cpu_buffer_out[i]=b+buffer_frames*(i+NUM_BUFFERS);
		gpu_buffer_out[i]=b+buffer_frames*(i+2*NUM_BUFFERS);
	}

	memset(b,0,sizeof(double)*buffer_frames*NUM_BUFFERS*3);

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

void init_audio() {
  int err;
 
  if ((err = snd_pcm_open (&capture_handle, CAMERA_USB_MIC_DEVICE, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    fprintf (stderr, "cannot open audio device(%s)\n", 
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "audio interface opened\n");
		   
  if ((err = snd_pcm_hw_params_malloc (&hw_params)) < 0) {
    fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params allocated\n");
				 
  if ((err = snd_pcm_hw_params_any (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params initialized\n");
	
  if ((err = snd_pcm_hw_params_set_access (capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
    fprintf (stderr, "cannot set access type (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params access setted\n");
	
  if ((err = snd_pcm_hw_params_set_format (capture_handle, hw_params, format)) < 0) {
    fprintf (stderr, "cannot set sample format (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params format setted\n");
	
  if ((err = snd_pcm_hw_params_set_rate_near (capture_handle, hw_params, &rate, 0)) < 0) {
    fprintf (stderr, "cannot set sample rate (%s)\n",
             snd_strerror (err));
    exit (1);
  }
	
  //fprintf(stdout, "hw_params rate setted\n");
 
  if ((err = snd_pcm_hw_params_set_channels (capture_handle, hw_params, 2)) < 0) {
    fprintf (stderr, "cannot set channel count (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params channels setted\n");
	
  if ((err = snd_pcm_hw_params (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot set parameters (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "hw_params setted\n");
	
  snd_pcm_hw_params_free (hw_params);
 
  //fprintf(stdout, "hw_params freed\n");
	
  if ((err = snd_pcm_prepare (capture_handle)) < 0) {
    fprintf (stderr, "cannot prepare audio interface for use (%s)\n",
             snd_strerror (err));
    exit (1);
  }
 
  //fprintf(stdout, "audio interface prepared\n");

}


unsigned Microseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec*1000000 + ts.tv_nsec/1000;
}

void init_fftw3() {
  p = fftw_plan_r2r_1d(buffer_frames, buffer_in[0], cpu_buffer_out[0] , FFTW_R2HC, FFTW_ESTIMATE);
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
		    if (exit_now==1) {	
			sem_post(&s_exit);
			return NULL;
		    }
	    }

            //just read in a half
	    half=1-half;
    }
}


void * process_audio(void * n) {
	//fprintf(stdout,"starting process_audio\n");
	int half=0;
	while (1) {
		//move from raw in to input
		int i;
		for (i=0; i<NUM_BUFFERS/2; i++) {
			sem_wait(&s_ready);
			if (exit_now==1) {
				sem_post(&s_done);
				sem_post(&s_exit);
				return NULL;
			}
		


			//COPY TO GPU		
			struct GPU_FFT_COMPLEX *gpu_base = gpu_fft->in+i*gpu_fft->step;
			int j;
			for (j=0; j<buffer_frames; j++) {
				gpu_base[j].re=raw_buffer_in[i+half*NUM_BUFFERS/2][j];
				gpu_base[j].im=0;
			}


			//COPY TO CPU
			short_to_double(buffer_in[i+half*NUM_BUFFERS/2],raw_buffer_in[i+half*NUM_BUFFERS/2],buffer_frames);
			
		}

		//unsigned t[4];

		//now we read in NUM_BUFFERS/2 chunks to GPU lets run this!
	    	//fprintf(stdout,"process_audio running gpu fft\n");
		//t[0]=Microseconds();
		gpu_fft_execute(gpu_fft); 
		//t[1]=Microseconds();
	    	/*fprintf(stdout,"process_audio running cpu fft\n");
		//t[2]=Microseconds();
		for (i=0; i<NUM_BUFFERS/2; i++) {
			fftw_execute_r2r(p,buffer_in[i+half*NUM_BUFFERS/2],cpu_buffer_out[i+half*NUM_BUFFERS/2]);
		}	
		//t[3]=Microseconds();

		//fprintf(stdout, "GPU %u vs CPU %u\n",t[1]-t[0],t[3]-t[2]);*/
	
		if (exit_now==1) {
			sem_post(&s_done);
			sem_post(&s_exit);
			return NULL;
		}

		//copy out GPU vallues
		for (i=0; i<NUM_BUFFERS/2; i++) {
			struct GPU_FFT_COMPLEX *gpu_base = gpu_fft->out+i*gpu_fft->step;
			int j;
			for (j=0; j<buffer_frames/2; j++) {
				gpu_buffer_out[i+half*NUM_BUFFERS/2][j]=gpu_base[j].re;
			}
			for (j=0; j<buffer_frames/2; j++) {
				gpu_buffer_out[i+half*NUM_BUFFERS/2][j+buffer_frames/2]=gpu_base[j+buffer_frames/2].im;
			}
		}


		for (i=0; i<NUM_BUFFERS/2; i++) {
			double d = logit(gpu_buffer_out[i+half*NUM_BUFFERS/2]);
			add_bark(d);
			//fprintf(stdout,"%f\n",sum_barks());
			if (sum_barks()<BARK_THRESHOLD) {
				time_t result = time(NULL);
				printf("BARK detected at %s\n", ctime(&result));
			}
			
		}

		//compare GPU and CPU on values
		/*for (i=0; i<NUM_BUFFERS/2; i++) {
			//lets find the difference between CPU and GPU computations
			double d =0.0;
			int j;
			for (j=0; j<buffer_frames; j++) {
				d+=fabs(gpu_buffer_out[i+half*NUM_BUFFERS/2][j]-cpu_buffer_out[i+half*NUM_BUFFERS/2][j]);	
			}	
			fprintf(stdout,"diff is %f\n", d);
		}*/
		
		sem_post(&s_done);
		if (exit_now==1) {
			sem_post(&s_exit);
			return NULL;
		}

		half=1-half;
	}
	
}


int main (int argc, char *argv[]) {
  assert(buffer_frames%2==0);
  fprintf(stdout,"reading model\n");

  read_model("./model");
  fprintf(stdout,"starting inits\n");
  init_barks();
  init_audio();
  init_buffers();
  init_gpu();
  init_fftw3();

  //set up the semaphores
  sem_init(&s_ready, 0, 0); 
  sem_init(&s_done, 0, 2); 


  //compute the output frequencies
  double * freq = (double*)malloc(sizeof(double)*buffer_frames);
  if (freq==NULL) {
 	fprintf(stderr, "Failed to alloc freq array\n");
	exit(1);
  }
	
  int i;
  for (i = 0; i < buffer_frames; i++) {
	freq[i]=(((double)i)/buffer_frames)*rate;
	fprintf(stdout, "%f%c" , freq[i], (i==buffer_frames-1) ?  '\n' : ',');
  }

  fprintf(stdout,"starting threads\n");

  pthread_t read_thread, process_thread;

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



  char line[2056];
  while (fgets(line, 2056, stdin)) {
	fprintf(stdout,"still here!\n");
  }
  fprintf(stdout,"Exiting!\n"); 
  exit_now=1;
 
  sem_wait(&s_exit);
  sem_wait(&s_exit);
	
  //wait for threads to exit
  gpu_fft_release(gpu_fft); // Videocore memory lost if not freed !
  fftw_destroy_plan(p);
  snd_pcm_close (capture_handle);

  for (i=0; i<NUM_BUFFERS; i++) {
	free(raw_buffer_in[i]);
	free(buffer_in[i]);
	free(cpu_buffer_out[i]);
	free(gpu_buffer_out[i]);
  }
  
  free_buffers();
  return 0;
  


}
