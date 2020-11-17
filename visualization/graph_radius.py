import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# kinetics
means_k = np.array([0.93534281, 0.95152063, 0.95972874, 0.9652154 , 0.96893228, 0.97359269, 0.97989329])
error_k = np.array([0.01648986, 0.01456212, 0.01267972, 0.01111384, 0.01010197, 0.00885579, 0.00701923])

# hollywood2
means_h = np.array([np.nan, np.nan, 0.8962, 0.9147, 0.9280, 0.9419, 0.9608])
error_h = np.array([np.nan, np.nan, 0.0319, 0.0283, 0.0238, 0.0194, 0.0136])

# finegym
means_f = np.array([0.9579, 0.9723, 0.9792, 0.9824, 0.9829, 0.9840, 0.9872])
error_f = np.array([0.0314, 0.0212, 0.0153, 0.0115, 0.0108, 0.0095, 0.0086])

# movienet
means_m = np.array([0.8544, 0.8942, 0.9120, 0.9195, 0.9253, 0.9368, 0.9454])
error_m = np.array([0.0555, 0.0502, 0.0436, 0.0398, 0.0405, 0.0337, 0.0315])

# finegym

fig, ax = plt.subplots(dpi=400)
fig.set_figheight(3.5)
# fig.set_figwidth(10)

# ax.bar(range(7), means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)


list_colors = ['royalblue', 'red', 'goldenrod', 'lightsalmon'] #, 'lightcoral', 'chocolate', 'lightsalmon', 'darkkhaki', 'palegreen', 'mediumturquoise', 'dodgerblue', 'indigo', 'deeppink']

color1 = "#003f5c"
color2 = "#7a5195"
color3 = "#ef5675"
color4 = "#ffa600"


plt1, = plt.plot([f't-{i}' for i in reversed(range(1, 7))] + ['t', ], means_k, 'k-', color=color1)   #color='#1B2ACC')
plt.fill_between(range(7), means_k-error_k, means_k+error_k, alpha=0.1, facecolor=color1 , zorder=0)  #'#089FFF')

plt2, = plt.plot([f't-{i}' for i in reversed(range(1, 7))] + ['t', ], means_m, 'k-', color=color2)  #color='#1B2ACC')
plt.fill_between(range(7), means_m-error_m, means_m+error_m, alpha=0.1, facecolor=color2, zorder=0)  #'#089FFF')

plt3, = plt.plot([f't-{i}' for i in reversed(range(1, 7))] + ['t', ], means_h, 'k-', color=color3)  #color='#1B2ACC')
plt.fill_between(range(7), means_h-error_h, means_h+error_h, alpha=0.1, facecolor=color3, zorder=0)  #'#089FFF')

plt4, = plt.plot([f't-{i}' for i in reversed(range(1, 7))] + ['t', ], means_f, 'k-', color=color4)  #color='#1B2ACC')
plt.fill_between(range(7), means_f-error_f, means_f+error_f, alpha=0.1, facecolor=color4, zorder=0)  #'#089FFF')

ax.set_ylabel('Poincaré ball radius', fontsize=15, labelpad=10)
ax.set_xlabel('Prediction time step', fontsize=15, labelpad=10)

ax.legend((plt1, plt2, plt3, plt4), ('Kinetics', 'MovieNet', 'Hollywood2', 'FineGym'))

#ax.set_title('Poincare ball radius evolution with time', fontsize=20)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)

ax.set_ylim(0.83, 1.)

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=False,
    labelbottom=True) # labels along the bottom edge are off

ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# for i in range(7):
# 	# ax.plot((0, 0), (ylim, 0.8297), linestyle='--', color='k', alpha=0.3, zorder=0)
# 	ax.plot((i, i), (0, means[i]), linestyle='--', color='k', alpha=0.3, zorder=0, linewidth=1)

# Save the figure and show
plt.tight_layout()
plt.savefig('/Users/didac/Desktop/figure_ruoshi.pdf')