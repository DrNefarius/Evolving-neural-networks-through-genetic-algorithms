import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# TODO: Get these times automatically!
times = [89.12476253509521, 94.35568881034851, 113.15996408462524, 108.75457906723022, 129.68387913703918,
         125.16838216781616, 174.1684229373932, 196.26968240737915, 193.54888534545898, 169.91287517547607,
         266.7449312210083, 343.8405783176422, 348.62483859062195, 508.0201849937439, 633.6475563049316]

gene_arr = range(1, constants.NGEN + 1)
plt.plot(gene_arr, times, 'bo-')
b_patch = mpatches.Patch(color='blue', label='TIME (s)')
plt.legend(handles=[b_patch])
plt.ylabel('TIME (s)')
plt.xlabel('GENERATION NUMBER')
plt.show()
plt.clf()
plt.cla()
plt.close()
