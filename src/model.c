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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double intercept;
double * w;
int model_size;

#define MAX_V_SIZE	4096


double logit(double * v) {
	double c = intercept;
	//fprintf(stdout,"%lf intercept\n",intercept);
	int i;
	double tmp[model_size];
	/*for (i=0; i<model_size; i++) {
		//fprintf(stdout, "%d %lf\n",i,w[i]);

		
		//regulat
		c+=w[i]*abs(v[i]);

		//fold back
		if (i<model_size/2) {
			c+=w[i]*(abs(v[i]) + abs(v[model_size-1-i]));
		}

		//fold back - with trim
		if (i<model_size/2) {
			if (i>=20 && i<150) {
				c+=w[i-20]*(abs(v[i]) + abs(v[model_size-1-i]));
			}
		}
	}*/


	//for (i=20; i<80; i++) {
	//	c+=w[i-20]*(abs(v[i]) + abs(v[model_size-1-i]));
	//}

	
	//reduce by half
	for (i=0; i<model_size; i++) {
		tmp[i]=fabs(v[i])+fabs(v[2*model_size-1-i]);
	}

	//blur and put back
	for (i=0; i<model_size; i++) {
		double cc=0;
		if (i>1 && i<((model_size)-2)) {
			cc =w[i]*(tmp[i-2]*0.1+tmp[i-1]*0.2+tmp[i]*0.4+tmp[i+1]*0.2+tmp[i+2]*0.1);
		}
		c+=cc;
	}

	/*for (i=0; i<model_size/2; i++) {
		//if (i>=3 || i<(model_size/2 - 3)) 
		if (i>1 && i<(model_size-2)) {
			c+=w[i]*(abs(v[i]) + abs(v[model_size-1-i]));
		} else {
			assert(abs(w[i])<0.00000000001);
		}
		//}
	}*/
	


	return 1.0/(1+exp(c));
}



void prepare_input(double ** b, int buffers, int length) {
	double * means = (double*)malloc(sizeof(double)*length);
	if (means==NULL) {
		fprintf(stderr,"Failed to malloc buffer!\n");
	}
	memset(means,0,sizeof(double)*length);

	//mean normalize
	int i,j;
	for (i=0; i<buffers; i++) {
		for (j=0; j<length; j++) {
			means[j]+=b[i][j];
		}
	}
	for (j=0; j<length; j++) {
		means[j]/=length;
	}
	for (i=0; i<buffers; i++) {
		for (j=0; j<length; j++) {
			b[i][j]-=means[j];
		}
	}
	free(means);	

	//foldback and abs
	for (i=0; i<buffers; i++) {
		for (j=0; j<length/2; j++) {
			b[i][j]=fabs(b[i][j])+fabs(b[i][length-1-j]);
		}
	}

	//drop half, but not really just skip it

	//blur
	for (i=0; i<buffers; i++) {
		for (j=2; j<length/4-2; j++) {
			b[i][j]=b[i][j-2]*0.1+b[i][j]*0.2+b[i][j]*0.4+b[i][j+1]*0.2+b[i][j]*0.1;	
		}
	}
	
	//mask
	for (i=0; i<buffers; i++) {
		b[i][0]=b[i][1]=b[i][2]=0;
	}

	//get the top 20 values
	int m;	
	for (m=0; m<20; m++) {
		for (i=0; i<buffers/4; i++) {
			double mx=0.0;
			int idx=0;
			for (j=0; j<length/2; j++) {
				if (b[i][j]>mx) {
					mx=b[i][j];	
					idx=j;
				}
			}
			b[i][idx]=-b[i][idx];
		}
	}
	for (i=0; i<buffers; i++) {
		for (j=0; j<length/4; j++) {
			if (b[i][j]>0) {
				b[i][j]=0;
			} else {
				b[i][j]=-b[i][j];
			}
		}
	}

	//take the log
	for (i=0; i<buffers; i++) {
		for (j=0; j<length/4; j++) {
			b[i][j]=log(b[i][j]+1);
		}
	}
		
}

int read_model(char * filename) {
	model_size=0;
	FILE * fptr = fopen(filename,"r");
	if (fptr==NULL) {
		fprintf(stderr, "Failed to open model %s\n", filename);
		exit(1);
	}

	w = (double*)malloc(sizeof(double)*MAX_V_SIZE);
	if (w==NULL) {
		fprintf(stderr, "Failed to alloc mem for mdel vec\n");	
		exit(1);
	}
	memset(w,0,sizeof(double)*MAX_V_SIZE); 

	char line[1024];
	if (fgets(line, 1024, fptr)) {
		int r = sscanf(line,  "%lf\n", &intercept);
		if (r!=1) {
			fprintf(stderr,"Failed to read header model\n");
			exit(1);
		}
	} else {
		fprintf(stderr,"Failed to read header model\n");
		exit(1);
	}
	while (fgets(line, 1024, fptr)) {
		int index;
		double value;
		int r =sscanf(line, "%d %lf\n",&index, &value);
		if (r!=2) {
			fprintf(stderr, "Failed to load model!\n");
			exit(1);
		}
	 	if (index>=MAX_V_SIZE) {
			fprintf(stderr, "Model is in apprpriate\n");
			exit(1);
		}
		w[index]=value;
		if ((index+1)>model_size) {
			model_size=index+1;
		}
	}
	fclose(fptr);
	return model_size;
}
