f = open("Data/URL.txt", "a")
#inp=input("Enter the url\n")
#inp+="\n"
import sys
	
print("Enter the data")
data = sys.stdin.read()   # Use Ctrl d to stop the input
data=list(data)
indices = [i for i, x in enumerate(data) if x == "\n"]
j=0
inp=input("enter label\n")
for i in indices:
	ted=i+j
	data.insert(ted,"\t"+inp)
	j+=1
f.write("".join(data))
