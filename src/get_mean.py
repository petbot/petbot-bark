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
        return m

data = read_file2(sys.argv[1])



d=[]
for v in data:
	for x in range(len(v)):
		while len(d)<=x:
			d.append([])
		d[x].append(v[x])

sm=[]
ss=[]
for x in range(len(d)):
	sm.append(str(mean(d[x])))
	ss.append(str(std(d[x])))

print ",".join(sm)
print ",".join(ss)

