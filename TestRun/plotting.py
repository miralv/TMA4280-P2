import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
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
#data = np.loadtxt("TestRun/g_P2_t1.out", usecols = 1)

cwd = os.getcwd()
files = glob.glob(cwd + "/TestRun/g_P2*.out")
k = np.arange(6,14,1)
n = 2**k
h = 1/n
np.savetxt("n.csv",n,delimiter=",",fmt="%i")

plt.loglog(h,h**2, '--', color='red',label = r"$\mathcal{O}(h^2)$")
#Since the errors are identical, it is sufficent to plot for one file
file = files[0]
#for i, file in enumerate(files):
files =  glob.glob(cwd + "/TestRun/g_P16_t2.out") 
file = files[0]
#file = files[3]
with open(file,"r") as infile:
    err = []
    counter = 1
    for line in infile:
        if ((counter%4)== 0):
            err.append(line.replace('\n','').split(":")[1])
        counter+=1

    #plot the current file
    #print(err)
    err = np.asfarray(err)
    np.savetxt("P16_t2.csv",err,delimiter=",")
    #plt.loglog(h,err, label = r"$P=2, t=1$")

plt.xlabel(r"$h$")
plt.ylabel(r"$||e_h||_\infty$")
plt.legend(loc = "center right")
plt.show()






print(err)
math.log10(n)
np.log10(n)
np.log10(err)
err = np.asfarray(err)
type(err[0])
print(n)
#plt.plot(np.log10(n),np.log10(err))
plt.loglog(h,err)
#plt.legend((r"$\mathcal{O}$"))
plt.xlabel(r"$\mathcal{O}$")
plt.show()
len(err)



# Save chosen files for putting in table 
np.savetxt("n.csv",n,delimiter=",",fmt="%i")




## Lastly: plot speedup!

T_1 = 1.251283e+03
P = [2,4,6,8,10,12]
T_P = [6.787136e+02,3.951214e+02, 2.710046e+02, 1.995009e+02,1.558085e+02,1.304740e+02] # Fyll inn manuelt ved å klippe og lime
S_P = np.divide(T_1,T_P)
par_eff = np.divide(S_P,P)
plt.plot(P,S_P, label = r"$S_p$")
# plot ideal speedup
plt.plot(P, P, '--', color='red', label = "$S_p = P$")
plt.xlabel("Processes ($P$)")
plt.ylabel("Speedup ($S_p$)")
plt.legend(loc = "lower right")
plt.show()

plt.plot(P,par_eff,'o')
plt.xlabel("Processes ($P$)")
plt.ylabel(r"Parallel efficiency ($\eta_p$)")

plt.show()