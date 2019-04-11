import matplotlib.pyplot as plt
import numpy as np
# First: Want to plot the timings for different number of processors
processors = [6,9,12,18,36]
timings = [4.485531e+01,5.029220e+01, 4.277237e+01 ,4.232179e+01,4.213946e+01]
labels = ["   $P=6, t=6$","   $P=9, t=4$","   $P=12, t=3$","   $P=18, t=2$","   $P=36, t=1$"]

len(timings)
len(labels)

fig,ax = plt.subplots()
ax.scatter(processors,timings)
for i, txt in enumerate(labels):
    ax.annotate(txt, (processors[i],timings[i]))
plt.xlim(5,40)
plt.xlabel("Processes ($P$)")
plt.ylabel("Time ($s$)")
plt.show()



# Second: Want to plot verification plots for a different number of P and t.

line = "e_max: 1.130315e-08"
res = line.find("e_max")
print(res)


f = open("TestRun/results/P_6.out","r")

value = np.loadtxt("TestRun/results/P_6.out", skiprows=12, usecols = 2)

if (f.mode == "r"):
    contents = f.read()
    res = contents.find("e_max")
    print(res)
    #for line in contents:
        #print(line)
        #print(line.find("e_max"))
        # if line.find("e_max")!=-1:
        #     splitted = line.split(":")
        #     #number = splitted[2]
        #     print(splitted)

#print(contents)
f.close()