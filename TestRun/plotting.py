import matplotlib.pyplot as plt
# First: Want to plot the timings for different number of processors



#6,9,18,36 processors
processors = [6,9,18,36]
timings = [2.415298e+02,1.851487e+02,8.187976e+01,4.171915e+01]
labels = ["   $P=6, t=6$","   $P=9, t=3$","   $P=18, t=2$","   $P=36, t=1$"]


sci = 2.415298e+02
sci = int(sci)
print(sci)
fig,ax = plt.subplots()
ax.scatter(processors,timings)
for i, txt in enumerate(labels):
    ax.annotate(txt, (processors[i],timings[i]+0.4))
plt.xlim(5,40)
plt.xlabel("Processes ($P$)")
plt.ylabel("Time ($s$)")
plt.show()



#plt.plot(processors,timings,'o')
#plt.show()