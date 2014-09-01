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
#include "model.h"

#define BUFFER_SIZE	1024*1024

void run_file(char * fn, int length) {
	FILE * fptr = fopen(fn,"r");
	if (fptr==NULL) {
		fprintf(stdout,"failed to open file %s\n",fn);
		exit(1);
	}	
	
	char * line = (char*)malloc(sizeof(char)*BUFFER_SIZE);
	if (line==NULL) {
		fprintf(stdout, "failed to malloc buffer\n");
		exit(1);
	}



	fgets(line,BUFFER_SIZE,fptr); //get rid of header
	while (fgets(line,BUFFER_SIZE,fptr)) {
		double v[length*2];
		//lets read in the v and then get the score
		char * c, *p;
		c=line;
		p=line;
		int i=0;
		while (*p!='\0') {
			while (*c!='\0' && *c!='\n' && *c!=',') {
				c++;
			}
			if (*c=='\n') {
				*c='\0';
			}

			char t=*c;
			*c='\0';
			if (i>=(2*length)) {
				fprintf(stdout,"failed to read in model, length is off! %d vs %d, %s\n",i,length,p);
				exit(1);
			}
			v[i++]=atof(p);
			//fprintf(stderr,"FLOAT %lf, %d\n",v[i-1], i);	
			if (t!='\0') {
				c++;
			}
			p=c;
		}
		fprintf(stdout,"%0.5lf\n",logit(v));
	}	
}

int main(int argc, char ** argv) {
	if (argc!=3) {
		fprintf(stdout, "%s model data\n",argv[0]);
		exit(1);
	}

	char * model_filename=argv[1];
	char * data_filename=argv[2];
	
	
	int length = read_model(model_filename);
	run_file(data_filename,length);
			
	return 0;
}
