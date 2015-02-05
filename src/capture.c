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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <alsa/asoundlib.h>

#define WINDOW_SIZE 256


#define CAMERA_USB_MIC_DEVICE "plughw:1,0"


struct WaveHeader {
        char RIFF_marker[4];
        uint32_t file_size;
        char filetype_header[4];
        char format_marker[4];
        uint32_t data_header_length;
        uint16_t format_type;
        uint16_t number_of_channels;
        uint32_t sample_rate;
        uint32_t bytes_per_second;
        uint16_t bytes_per_frame;
        uint16_t bits_per_sample;
};

struct WaveHeader *genericWAVHeader(uint32_t sample_rate, uint16_t bit_depth, uint16_t channels){
    struct WaveHeader *hdr;
    hdr = (struct WaveHeader*) malloc(sizeof(*hdr));
    if (!hdr)
        return NULL;

    memcpy(&hdr->RIFF_marker, "RIFF", 4);
    memcpy(&hdr->filetype_header, "WAVE", 4);
    memcpy(&hdr->format_marker, "fmt ", 4);
    hdr->data_header_length = 16;
    hdr->format_type = 1;
    hdr->number_of_channels = channels;
    hdr->sample_rate = sample_rate;
    hdr->bytes_per_second = sample_rate * channels * bit_depth / 8;
    hdr->bytes_per_frame = channels * bit_depth / 8;
    hdr->bits_per_sample = bit_depth;

    return hdr;
}

int writeWAVHeader(int fd, struct WaveHeader *hdr){
    if (!hdr)
        return -1;

    write(fd, &hdr->RIFF_marker, 4);
    write(fd, &hdr->file_size, 4);
    write(fd, &hdr->filetype_header, 4);
    write(fd, &hdr->format_marker, 4);
    write(fd, &hdr->data_header_length, 4);
    write(fd, &hdr->format_type, 2);
    write(fd, &hdr->number_of_channels, 2);
    write(fd, &hdr->sample_rate, 4);
    write(fd, &hdr->bytes_per_second, 4);
    write(fd, &hdr->bytes_per_frame, 2);
    write(fd, &hdr->bits_per_sample, 2);
    write(fd, "data", 4);

    uint32_t data_size = hdr->file_size + 8 - 44;
    write(fd, &data_size, 4);

    return 0;
}	      



void ShortToReal(signed short* shrt,double* real,int siz) {
	int i;
	for(i = 0; i < siz; ++i) {
		real[i] = shrt[i]; // 32768.0;
	}
}



signed short *buffer;
int buffer_frames = 2048*4*16;
unsigned int rate = 8000;
double *buffer_out, *buffer_in, *power_spectrum;
snd_pcm_t *capture_handle;
snd_pcm_hw_params_t *hw_params;
snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

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

int main (int argc, char *argv[]) {
	
  if (argc!=2 ) {
	fprintf(stderr,"%s filename\n",argv[0]);
	exit(1);
  }
  assert(buffer_frames%2==0);


 char * fn = argv[1];
  struct WaveHeader *hdr;
  hdr = genericWAVHeader(8000, 16, 2);


  init_audio();
  buffer = malloc(buffer_frames * snd_pcm_format_width(format) / 8 * 2);
  buffer_in = malloc(buffer_frames * sizeof(double));

  int err;
    if ((err = snd_pcm_readi (capture_handle, buffer, buffer_frames)) != buffer_frames) {
      fprintf (stderr, "read from audio interface failed (%s)\n",
               snd_strerror (err));
      exit (1);
    }
    //convert to frequency domain

    //cast over to double
    ShortToReal(buffer,buffer_in,buffer_frames);


  fprintf(stderr,"captured %d points\n",err);
  uint32_t pcm_data_size = hdr->bytes_per_frame * err;
  hdr->file_size = pcm_data_size + 44 - 8;

 //now write the file
  int filedesc = open(fn, O_WRONLY | O_CREAT, 0644);
  err = writeWAVHeader(filedesc, hdr);
    if (err) {
		fprintf(stderr,"Failed to write header\n");
		exit(1);
    }

  write(filedesc, buffer, pcm_data_size);
  exit(1);

  
  int i;
 




  buffer_out = malloc(buffer_frames * sizeof(double));
  power_spectrum = malloc(buffer_frames * sizeof(double));

  if (buffer==NULL || buffer_in==NULL || buffer_out==NULL || power_spectrum==NULL) {
	fprintf(stderr, "Failed to allocate memory for buffers\n");
	exit(1);
  }


  /*fftw2*/
  //rfftw_plan p;
  //p = rfftw_create_plan(buffer_frames, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
  /*fftw3*/
  fftw_plan p;
  p = fftw_plan_r2r_1d(buffer_frames, buffer_in, buffer_out , FFTW_R2HC, FFTW_ESTIMATE);


  //compute the output frequencies
  double * freq = (double*)malloc(sizeof(double)*buffer_frames);
  if (freq==NULL) {
 	fprintf(stderr, "Failed to alloc freq array\n");
	exit(1);
  }
  //for (i = 0; i < buffer_frames; i++) {
  //	freq[i]=(((double)i)/buffer_frames)*rate;
  //	fprintf(stdout, "%f%c" , freq[i], (i==buffer_frames-1) ?  '\n' : ',');
  //}

  
  //int err;
  if ((err = snd_pcm_readi (capture_handle, buffer, buffer_frames)) != buffer_frames) {
     fprintf (stderr, "read from audio interface failed (%s)\n",
               snd_strerror (err));
     exit (1);
   }
    //convert to frequency domain

    //cast over to double
    ShortToReal(buffer,buffer_in,buffer_frames);
    //print the transformed data
   int j;
    for (j=0; j<buffer_frames; j++) {
	fprintf(stdout, "%f%c" ,buffer_in[j],(j==buffer_frames-1) ? '\n' : ',');
    }
    exit(1);
	
  //get a bunch more
  for (i = 0; i < 800; ++i) {
    if (i%100==0) {
	fprintf(stderr,"%d\n",i);
    }
    if ((err = snd_pcm_readi (capture_handle, buffer, buffer_frames)) != buffer_frames) {
      fprintf (stderr, "read from audio interface failed (%s)\n",
               snd_strerror (err));
      exit (1);
    }
    //convert to frequency domain

    //cast over to double
    ShortToReal(buffer,buffer_in,buffer_frames);
    //clear buffers 
    memset(buffer_out, 0, sizeof(double)*buffer_frames);
    memset(power_spectrum, 0, sizeof(double)*buffer_frames);

    //run the fft    
    /*int y; 
    for (y=0; y<buffer_frames; y++) {
	buffer_in[y]=y;
    }*/
    fftw_execute(p);

    //power_spectrum
    /*
Here, rkis the real part of the kth output, and ikis the imaginary part. (Division by 2 is rounded down.) For a halfcomplex array hc[n], the kth component thus has its real part in hc[k] and its imaginary part in hc[n-k], with the exception of k == 0 or n/2 (the latter only if n is even)—in these two cases, the imaginary part is zero due to symmetries of the real-input DFT, and is not stored. Thus, the r2hc transform of n real values is a halfcomplex array of length n, and vice versa for hc2r.
    */
    /* from fftw2
An FFTW_FORWARD transform corresponds to a sign of -1 in the exponent of the DFT. Note also that we use the standard "in-order" output ordering--the k-th output corresponds to the frequency k/n (or k/T, where T is your total sampling period). For those who like to think in terms of positive and negative frequencies, this means that the positive frequencies are stored in the first half of the output and the negative frequencies are stored in backwards order in the second half of the output. (The frequency -k/n is the same as the frequency (n-k)/n.)
    */

    int j;
	
    //compute the power spectrum by adding the real and imaginary ? 
    /*
    power_spectrum[0] = 0 ; //buffer_out[0]*buffer_out[0];
    int k;
    for (k = 1; k < (buffer_frames+1)/2; ++k)
	power_spectrum[k] = buffer_out[k]*buffer_out[k] + buffer_out[buffer_frames-k]*buffer_out[buffer_frames-k];
    //if (buffer_frames % 2 == 0) //TRUE BY ASSERTION
    power_spectrum[buffer_frames/2] = buffer_out[buffer_frames/2]*buffer_out[buffer_frames/2];  // Nyquist freq.
    for (j=0; j<buffer_frames/2; j++) {
	fprintf(stdout, "%f, " , power_spectrum[j]);
    }
    fprintf(stdout, "%f\n", power_spectrum[j]); */


    //print the transformed data
    for (j=0; j<buffer_frames; j++) {
	fprintf(stdout, "%f%c" ,buffer_out[j],(j==buffer_frames-1) ? '\n' : ',');
    }

  }
 
  fftw_destroy_plan(p);
  free(buffer);
 
	
  snd_pcm_close (capture_handle);

  return 0; 
}
