import alsaaudio, wave, numpy

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, 'Camera')
inp.setchannels(1)
inp.setrate(8000)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(1024*8*8*8*8)

w = wave.open('test.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)

while True:
    l, data = inp.read()
    a = numpy.fromstring(data, dtype='int16')
    print numpy.abs(a).mean()
    w.writeframes(data)
