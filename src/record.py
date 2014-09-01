#This file is part of PetBot.
#
#    PetBot is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PetBot is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PetBot software.  If not, see <http://www.gnu.org/licenses/>.

import alsaaudio, wave, numpy
import sys

from time import sleep
import numpy as np

chunk=471
sample_rate=44100
channels=1

#print alsaaudio.cards()

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, 'Camera')
inp.setchannels(channels)
inp.setrate(sample_rate)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(chunk)


outfile="xA"

data=[]

def process_data(l,datastr,dump):
    a = numpy.fromstring(datastr, dtype='int16')
    #print len(a)
    assert(len(a)==chunk)
    fourier=np.fft.rfft(a)
    d=fourier.real 
    '''s=0
    i=0
    for x in range(len(d)):
         if freq[x]>400 and freq[x]<700:
		s+=d[x]
		i+=1.0
    return np.mean(fourier.real), s/i'''
    #out=open(outprefix+str(dump)+".txt",'w')
    #for x in range(1,len(d)-1):
    #     print >> out, freq[x], d[x]
    #out.close()
    print len(d)
    sys.exit(1)
    data.append(d)
    dump+=1
    print "OK",dump
    


freq = np.fft.rfftfreq(chunk, 1/float(sample_rate))	
print len(freq)
dump=0

while dump<-1: #<4000:
    l, datastr = inp.read()
    #pass off to worker thread?
    if l>0:
	dump+=1
    	process_data(l,datastr,dump)
	sleep(0.1)
    #sys.exit(1)
    #a = numpy.fromstring(data, dtype='int16')
    #print numpy.abs(a).mean()
    #w.writeframes(data)

h=open(outfile,'w')
for x in freq:
	print >> h, x
for x in data:
	for y in x:
		print >> h, y
h.close()
