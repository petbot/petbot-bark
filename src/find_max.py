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
from numpy import mean , std

def read_file2(fn):
        h=open(fn) 
        m=[]  
        first_line=True
        freq=[]
        for line in h:
                if first_line:
                        freq=map(lambda x : float(x) , line.strip().split(',')  )
                        first_line=False
                else:
                        v= map(lambda x : abs(float(x)) , line.strip().split(',')  )
                        m.append(v)

        h.close()
        return freq,m

freq,data = read_file2(sys.argv[1])


def fold_back(v):
        v2=[]
        for x in range(len(v)/2):
                v2.append( abs(v[x]) +abs(v[len(v)-x-1]))
         
        #v2=v2[3:-3]
         
        for x in range(2):
                v2[x]=0
        for x in range(512):
                v2[-x]=0
   
        #v2=v2[20:80]
        return v2

maxes=20

mxs=[]
for v in data:
	m=[]
	v=fold_back(v)
	for x in range(maxes):
		mx=max(v)
		m.append(v.index(mx))
		v[v.index(mx)]=0
	mxs.append(m)	

print ",".join(map(lambda x : str(x), range(maxes)))
for v in mxs:
	print ",".join(map(lambda x : str(x), v))
