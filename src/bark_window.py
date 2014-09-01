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





if len(sys.argv)!=4:
	print "%s window sum file" % sys.argv[0]
	sys.exit(1)


window_size=int(sys.argv[1])
bark_sum=float(sys.argv[2])
fn=sys.argv[3]


s=[]
b=0
h=open(fn)
for line in h:
	s.append(float(line))
	if len(s)>window_size:
		s=s[1:]
	if len(s)==window_size:
		if sum(s)<bark_sum:
			b+=1
h.close() 		
print b
