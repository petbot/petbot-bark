gcc -o listen_for_bark -lasound -lfftw3 -lm  -lrt ./listen_for_bark_gpu.c ./model.c -O3  -Wall  mailbox.c gpu_fft.c gpu_fft_twiddles.c gpu_fft_shaders.c -lpthread
#gcc -o listen_for_bark -lasound -lfftw3 -lm ./listen_for_bark.c ./model.c -O3  -Wall
#gcc -o run_model -lasound -lfftw3 -lm run_model.c ./model.c -Wall -O3 -Wall 
#gcc -o capture -lasound -lfftw3 -lm ./capture.c -O3
