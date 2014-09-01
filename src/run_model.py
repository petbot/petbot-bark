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

import sys
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from numpy import mean, std
from sklearn.decomposition import PCA
from math import exp


def fold_back(v):
	v2=[]
	for x in range(len(v)/2):
		v2.append( abs(v[x]) +abs(v[len(v)-x-1]))
	
	#smoothing
	v3=[]
	for x in range(len(v2)):
		if x>1 and x<len(v2)-2:
			v3.append(v2[x-2]*0.1+v2[x-1]*0.2+v2[x]*0.4+v2[x+1]*0.2+v2[x+2]*0.1)
		else:
			v3.append(v2[x])
	v2=v3
	for x in range(3):
		v2[x]=0
	for x in range(512):
		v2[-x]=0


			

	#v2=v2[20:80]
	return v2 	

def read_file2(fn):
	h=open(fn)
	m=[]
	first_line=True
	freq=[]
	for line in h:
		if first_line:
			freq=map(lambda x : float(x) , line.strip().split(',')	)
			first_line=False
		else:
			v= map(lambda x : abs(float(x)) , line.strip().split(',')  )
			v=fold_back(v)

			#for x in range(len(v)):
			#	if v[x]<3000:
			#		v[x]=0
			m.append(v)
	
	h.close()
	return m


data=[]
model_file=sys.argv[1]
for fn in sys.argv[2:]:
	d=read_file2(fn)
	data+=d



def read_model(fn):
	h=open(fn,'r')
	lines=map(lambda x : x.strip(), h.readlines())
	intercept=float(lines[0])
	w=[]
	for x in lines[1:]:
		w.append(float(x.split()[1]))
	h.close()
	return intercept,w

def logit(i,w,v):
	c=[i]
	assert(len(w)==len(v))
	for x in range(len(w)):
		c.append(w[x]*abs(v[x]))
	return 1/(1+exp(sum(c)))

def logitc(i,w,v):
	c=i
	assert(len(w)==len(v))
	for x in range(len(w)):
		c+=w[x]*abs(v[x])
	return c

#print clf.predict(data)
intercept,w=read_model(model_file)
for x in range(len(data)):
	print "%0.5f" % logit(intercept,w,data[x])
