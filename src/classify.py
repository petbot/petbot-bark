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
import numpy as np
from sklearn.decomposition import PCA
from math import exp
from math import log
from random import random
from multiprocessing import Pool

penalty="l1"
save_file="modelU_blur_"+penalty

def rand_shift(d,r1=0.2):
	if len(d)==0:
		return d
	l=len(d[0])/2
	assert(l%2==0)
	d2=np.zeros((d.shape[0],l))
	for i in xrange(len(d)):
		for x in xrange(len(d[i])):
			r=random()
			if r<r1 and x>1 and x<len(d[i])-1:
				r=random()
				if r>0.5:
					#do nothin
					d2[i][x]=d[i][x]
				else:
					if r>0.25:
						#go right
						d2[i][x+1]=d[i][x]
					else:
						#go left
						d2[i][x-1]=d[i][x]
						d2[i][x]=d[i][x-1]
			else:
				if d2[i][x]>0:
					d2[i][x-1]=d[i][x]
				else:
					d2[i][x]=d[i][x]

def fold_back(d):
	if len(d)==0:
		return d
	l=len(d[0])/2
	assert(l%2==0)
	d2=np.zeros((d.shape[0],l))
	for i in xrange(len(d)):
		d2[i]=d[i][:l]+d[i][::-1][:l]
	return d2

def blur(d):
	if len(d)==0:
		return d
	l=len(d[0])
	d2=np.zeros((d.shape[0],l))
	for i in xrange(len(d)):
		for x in xrange(len(d[i])):
			if x>1 and x<len(d[i])-2:
				d2[i]=d[i][x-2]*0.1+d[i][x-1]*0.2+d[i][x]*0.4+d[i][x+1]*0.2+d[i][x+2]*0.1
			else:
				d2[i]=d[i]
	return d2
		
def mask(d,n=3,ln=0):
	if len(d)==0:
		return d
	l=len(d[0])
	d2=np.zeros((d.shape[0],l))
	for i in xrange(len(d)):
		d2[i]=d[i][:]
		d2[i,:n]=0
		d2[i,:-ln]=0
	return d2

def drop_half(d,second_half=True):
	if len(d)==0:
		return d
	l=len(d[0])
	d2=np.zeros((d.shape[0],l/2))
	for i in xrange(len(d)):
		if second_half:
			d2[i]=d[i][:l/2]
		else:
			d2[i]=d[i][l/2:]
	return d2

def tops(d,t=20):			
	if len(d)==0:
		return d
	l=len(d[0])
	d2=np.zeros((d.shape[0],l))
	for i in xrange(len(d)):
		for x in xrange(t):
			j=np.argmax(d[i])
			d2[i][j]=d[i][j]
			d[i][j]=0
	return d2
		
def read_file(fn):
	try:
		m=np.load(fn+'.dump.npy')
		print "Loaded pickle file... ", fn
		return m
	except:
		pass
	
	
	h=open(fn)
	lines=h.readlines()
	freq=map(lambda x : float(x) , lines[0].strip().split(','))
	m=np.zeros((len(lines)-1,len(freq)))
	freq=[]
	i=0
	for line in lines[1:]:
		m[i]=np.fromstring(line,sep=',')
		i+=1
	h.close()
	np.save(fn+".dump",m)
	return m

def normalize(d):
	d2=np.copy(d)
	for x in xrange(len(d2)/10):
		m=mean(d2[x*10:(x+1)*10],0)
		#sd=std(d[x*10:(x+1)*10],0)+1
		d2[x*10:(x+1)*10]-=m
		#d2[x*10:(x+1)*10]/=m
		#d[x*10:(x+1)*10]/=sd
	return d2

def filter_uncommon(d,m=0.03):
	if len(d)==0:
		return d
	l=len(d[0])
	t=sum(d>0,0)
	ns=[]
	for x in xrange(len(t)):
		if t[x]>len(d)*m:
			ns.append(x)
	d2=np.zeros((d.shape[0],len(ns)))
	for x in xrange(len(ns)):
		d2[:,x]=d[:,ns[x]]
	print "Dropping %d and keeping %d" % (l-len(ns),len(ns))
	return d2,{'selected':ns,'length':l}

negatives=None
positives=None


def read_and_process(fn):
	d=normalize(read_file(fn))
	d=np.absolute(d)
	d=fold_back(d)
	d=blur(d)
	d=mask(d)
	d=drop_half(d)
	d=tops(d)
	#d=blur(d)
	#d=rand_shift(d)
	print "Loaded %d from %s" % (len(d), fn)
	return d

fns=sys.argv[1:]
p = Pool()
ds = p.map(read_and_process, fns)
#ds=map(read_and_process,fns)

for x in xrange(len(fns)):
	fn=fns[x]
	if fn.find('dog')>=0:
		if positives==None:
			positives=ds[x]
		else:
			positives=np.append(positives,ds[x],axis=0)
	else:
		if negatives==None:
			negatives=ds[x]
		else:
			negatives=np.append(negatives,ds[x],axis=0)



data=np.append(negatives,positives,axis=0)
raw_data=data.copy()
data,filtered=filter_uncommon(data,m=0.01)
#data=raw_data
#filtered={'selected':range(512),'length':512}

#print len(negatives),len(positives), len(data)
labels=[0]*len(negatives)+[1]*len(positives)

#data = preprocessing.scale(data)

test_data=[]
test_labels=[]
train_data=[]
train_labels=[]
for x in range(len(data)):
	if x%2==0:
		test_data.append(data[x])
		test_labels.append(labels[x])
	else:
		train_data.append(data[x])
		train_labels.append(labels[x])

#pca = PCA(n_components=10)
#data = pca.fit(train_data).transform(data)

def print_stats(d,l,s,clf,f=None):
	tp=0
	tn=0
	fn=0
	fp=0
	for x in range(len(d)):
		c=0
		if f==None:
			c=clf.predict(d[x])
		else:
			c=f(clf,d[x])<0.5
		if l[x]==1:
			if c==1:
				tp+=1
			else:
				fn+=1	
		else:
			if c==1:
				fp+=1
			else:
				tn+=1
	print "%s - FP:\t%10d,\tTN:\t%10d,\tTP:\t%10d,\tFN:\t%10d" % (s,fp,tn,tp,fn)
	

train=1
if train==1:
	print "Training model..."
	#neigh = KNeighborsClassifier(n_neighbors=10)
	#neigh.fit(train_data, train_labels) 
	#clf = NearestCentroid()
	#clf=svm.SVC()#kernel='linear')#class_weight='auto')
	#clf=svm.LinearSVC(class_weight='auto')
	#clf = tree.DecisionTreeClassifier()
	clf = LogisticRegression(penalty=penalty)
	#clf = RandomizedLogisticRegression()
	#clf = LogisticRegression()
	clf.fit(train_data, train_labels) 
	#clf.sparsify()
	#print clf.tree_
	#with open("output.dot", "w") as output_file:
	#    tree.export_graphviz(clf, out_file=output_file)
	print "Evaluating model..."
	print_stats(test_data,test_labels,"TEST",clf)
	print_stats(train_data,train_labels,"TRAIN",clf)
	i=0
	a = clf.coef_
	save_fn=save_file
	save_f=open(save_fn,'w')
	print >> save_f, clf.intercept_[0]
	for x in xrange(filtered['length']):
		print >> save_f, x,
		if x in filtered['selected']:
			i=filtered['selected'].index(x)
			print >> save_f, a[0][i]
		else:
			print >> save_f, 0
	save_f.close()

def read_model(fn):
	h=open(fn,'r')
	lines=map(lambda x : x.strip(), h.readlines())
	intercept=float(lines[0])
	w=[]
	for x in lines[1:]:
		w.append(float(x.split()[1]))
	h.close()
	return (intercept,w)

def logit((i,w),v):
	c=i
	assert(len(w)==len(v))
	for x in range(len(w)):
		c+=w[x]*abs(v[x])
	return 1/(1+exp(c))


clf=read_model(save_file)
print_stats(raw_data,labels,"ALL",clf,f=logit)



