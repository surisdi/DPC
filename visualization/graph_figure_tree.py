import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib._color_data as mcd

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

input_linear = torch.load('/Users/didac/proj/didac/results/input_linear.pth')
labels = torch.load('/Users/didac/proj/didac/results/labels.pth')

representations = input_linear.pow(2).sum(-1).sqrt().view(150, 6)[labels[:, 0, -1] != -1]

# representations = torch.tensor([[[0.7, 0.7], [0.75, 0.75],  [0.8, 0.8],  [0.85, 0.85],  [0.87, 0.87],  [0.9, 0.9]]])

means = representations.mean(0).cpu().numpy()
error = representations.std(0).cpu().numpy()
fig, ax = plt.subplots(dpi=400)
fig.set_figheight(2.5)
fig.set_figwidth(10)


#ax.axhline(linewidth=3, color=(255/255., 0/255., 0/255., 0.9), y=np.percentile(representations.cpu().numpy(), 66), linestyle='-.', zorder=0)
#ax.axhline(linewidth=3, color=(255/255., 127/255., 14/255., 0.9), y=np.percentile(representations.cpu().numpy(), 33), linestyle='-.', zorder=0)
ax.axhline(linewidth=3, color=(128/255., 0/255., 200/255., 0.9), y=np.percentile(representations.cpu().numpy(), 66), linestyle=':', zorder=0)
ax.axhline(linewidth=3, color=(128/255., 0/255., 200/255., 0.9), y=np.percentile(representations.cpu().numpy(), 33), linestyle='--', zorder=0)

# ax.bar(range(6), means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

# plt.plot([f't-{i}' for i in reversed(range(1, 6))] + ['t', ], means, 'k-', color=(0/255., 0/255., 255/255, 0))  #color='#1B2ACC')
# plt.fill_between(range(6), means-error, means+error, alpha=0.1, facecolor=(0/255., 0/255., 255/255,), zorder=0)  #'#089FFF')
ax.get_xaxis().set_visible(False)

# ax.set_ylabel('Poincare ball radius', fontsize=18, labelpad=10)
# ax.set_title('Poincare ball radius evolution with time', fontsize=20)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ylim = 0.63

ax.set_ylim(ylim+0.1, 1.01)

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=False,
    labelbottom=True) # labels along the bottom edge are off

ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

size_ball = 130
# ax.plot((0, 0), (ylim, 0.8297), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((0, 0), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([0], [0.8297], color='#ff57f0ff', s=size_ball)

# ax.plot((1, 1), (ylim, 0.8712), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((1, 1), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([1], [0.8712], color='#57b8ffff', s=size_ball)

# plt.plot((2, 2), (ylim, 0.9097), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((2, 2), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([2], [0.9097], color='#57b8ffff', s=size_ball)

# plt.plot((3, 3), (ylim, 0.9272), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((3, 3), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([3], [0.9272], color='#ff5757ff', s=size_ball)

# plt.plot((4, 4), (ylim, 0.9417), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((4, 4), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([4], [0.9417], color='#57ff86ff', s=size_ball)

# plt.plot((5, 5), (ylim, 0.9509), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.plot((5, 5), (ylim, 1.1), linestyle='--', color='k', alpha=0.3, zorder=0)
ax.scatter([5], [0.9509], color='#57ff86ff', s=size_ball)

head_width = 0#0.01
head_length = 0#0.04
margin = 0  #0.005
marginx = 0 #0.1
plt.arrow(0, 0.8297, 1-marginx, 0.8712-0.8297, head_width=head_width, head_length=head_length, zorder=0, color='k')
plt.arrow(1, 0.8712, 1-marginx, 0.9097-0.8712, head_width=head_width, head_length=head_length, zorder=0, color='k')
plt.arrow(2, 0.9097, 1-marginx, 0.9272-0.9097, head_width=head_width, head_length=head_length, zorder=0, color='k')
plt.arrow(3, 0.9272, 1-marginx, 0.9417-0.9272, head_width=head_width, head_length=head_length, zorder=0, color='k')
plt.arrow(4, 0.9417, 1-marginx, 0.9509-0.9417, head_width=head_width, head_length=head_length, zorder=0, color='k')

# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

plt.gca().invert_yaxis()
# Save the figure and show
plt.tight_layout()
plt.savefig('/Users/didac/Desktop/figure.pdf')