#PetBot listen

An open source repository for [PetBot](http://petbot.ca) bark recognition software.

Here we store code to recognize dog barks from a microphone on the raspberry pi.

Included in this repository is:

* **src/listen_for_bark.c** - The main executable, listens over ALSA and runs the LR model in realtime
* **src/run_model.c** - Load LR model and run on output from capture
* **src/capture.c** - Listen on microphone and collect data already passed through FFT
* **src/classify.py** - Train a LR (Logistic regression) model on output of capture
* **model/** - Contains trained LR models

Run by using:

``` ./listen_for_bark ```

## License
All content here is copyright Michael (Misko) Dzamba 2014. Unless otherwise stated in the headers. Please feel free to use any of my code or trained models for any personal projects. If you would like to package parts of this software with your product please contact me for further details.
