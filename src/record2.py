import alsaaudio, wave, numpy

import sys

if len(sys.argv)!=3:
	print "%s out_fn seconds" % sys.argv[0]
	sys.exit(1)

seconds=int(sys.argv[2])
out_wav_fn=sys.argv[1]

fs=8000

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, 'Camera')
inp.setchannels(1)
inp.setrate(fs)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
to_capt=fs*seconds
inp.setperiodsize(8000)


w = wave.open(out_wav_fn, 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)

while to_capt>0:
	l, data = inp.read()
	to_capt-=l
	a = numpy.fromstring(data, dtype='int16')
	#print numpy.abs(a).mean()
	w.writeframes(data)
