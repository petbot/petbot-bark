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
from sklearn import tree
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from numpy import mean, std
from sklearn.decomposition import PCA


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
			m.append( map(lambda x : abs(float(x)) , line.strip().split(',')  ) )
	h.close()
	return m


#if len(sys.argv)!=3:
#	print "%s negatives positives" % sys.argv[0]
#	sys.exit(1)


data=[]

cl=sys.argv[1]
train_fn=sys.argv[2]
test_fn=sys.argv[3]
for fn in sys.argv[4:]:
	d=read_file2(fn)
	#print "Loaded %d from %s" % (len(d), fn)
	data+=d


train_f=open(train_fn,'w')
test_f=open(test_fn,'w')

for i in range(len(data)):
	v=data[i]
	o=cl+"   "
	for x in range(len(v)):
		o=o+"%d:%f " % (x,v[x])
	if i%2==0:
		print >> test_f, o 
	else:
		print >> train_f, o
	

