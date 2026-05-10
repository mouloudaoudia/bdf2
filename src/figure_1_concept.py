import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'mathtext.fontset': 'dejavusans',
})

# Coefficient functions from the paper
# c1(theta) = (1 - 3 theta)/2
# c2(theta) = (1 + 2 theta - 3 theta^2)/4 in absolute value plot

theta = np.linspace(0.001, 0.999, 800)
c1 = np.abs((1 - 3 * theta) / 2)
c2 = np.abs((1 + 2 * theta - 3 * theta**2) / 4)

fig = plt.figure(figsize=(13.4, 5.6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ---------------- Panel A: schematic ----------------
t_star = 0.56
h = 0.22
theta_demo = 0.42
tk = t_star - theta_demo * h
tk_minus_1 = tk - h
tk_plus_1 = tk + h

xL = np.linspace(0.05, t_star, 300)
yL = 0.60 + 0.11*np.sin(5*xL) + 0.12*xL
xR = np.linspace(t_star, 0.98, 300)
yR = yL[-1] + 0.08 + 0.16*(xR - t_star) + 0.025*np.sin(7*xR + 0.5)

ax1.plot(xL, yL, lw=2.5)
ax1.plot(xR, yR, lw=2.5)
ax1.axvline(t_star, ls='--', lw=1.5, color='k')

# lower time axis and nodes
axis_y = 0.10
ax1.plot([0.18, 0.82], [axis_y, axis_y], color='0.4', lw=1.1)
for x, label in [(tk_minus_1, r'$t_{k-1}$'), (tk, r'$t_k$'), (tk_plus_1, r'$t_{k+1}$')]:
    ax1.plot([x, x], [axis_y-0.01, axis_y+0.11], color='0.45', lw=1.2)
    ax1.plot([x], [axis_y+0.015], 'o', ms=5.5, color='0.25')
    ax1.text(x, axis_y-0.05, label, ha='center', va='top')

ax1.text(t_star, 1.03, r'$t^{\star}$', ha='center', va='bottom', fontsize=12)

# spacing annotations
ax1.annotate('', xy=(tk, axis_y+0.06), xytext=(t_star, axis_y+0.06),
             arrowprops=dict(arrowstyle='<->', lw=1.2))
ax1.text((tk+t_star)/2, axis_y+0.09, r'$\theta h$', ha='center')

ax1.annotate('', xy=(t_star, axis_y+0.14), xytext=(tk_plus_1, axis_y+0.14),
             arrowprops=dict(arrowstyle='<->', lw=1.2))
ax1.text((t_star+tk_plus_1)/2, axis_y+0.17, r'$(1-\theta)h$', ha='center')

# upper stencil arrows
for x0, x1 in [(tk_minus_1, tk), (tk, tk_plus_1)]:
    ax1.add_patch(FancyArrowPatch((x0, 0.90), (x1, 0.90),
                                  arrowstyle='-|>', mutation_scale=12,
                                  lw=1.35, color='0.25'))
ax1.text((tk_minus_1+tk_plus_1)/2, 0.94, 'BDF2 crossing stencil', ha='center')

# labels
ax1.text(0.16, 0.72, 'left jet')
ax1.text(0.76, 0.80, 'right jet')
ax1.text(t_star+0.02, 0.60, r'jumps $J_1,\,J_2$', fontsize=12)
ax1.set_title('A. Mixed-side crossing geometry', loc='left', fontweight='bold')
ax1.set_xlim(0.02, 1.02)
ax1.set_ylim(0.0, 1.10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)

# ---------------- Panel B: sensitivity ----------------
ax2.plot(theta, c1, lw=2.4,
         label=r'$|c_1(\theta)|=\left|\frac{1-3\theta}{2}\right|$')
ax2.plot(theta, c2, lw=2.4,
         label=r'$|c_2(\theta)|=\left|\frac{1+2\theta-3\theta^2}{4}\right|$')
ax2.axvline(1/3, ls='--', lw=1.4, color='0.35')
ax2.annotate(r'$\theta=\frac{1}{3}$', xy=(1/3, 0.02), xytext=(0.41, 0.18),
             arrowprops=dict(arrowstyle='->', lw=1.1), fontsize=11)
ax2.text(0.63, 0.43,
         r'severe: $R^{\mathrm{raw}}_{k+1}\sim c_1(\theta)J_1$' + '\n' + r'generically $\mathcal{O}(1)$',
         fontsize=11)
ax2.text(0.56, 0.14,
         r'benign: $R^{\mathrm{raw}}_{k+1}\sim -h\,c_2(\theta)J_2$' + '\n' + r'$\mathcal{O}(h)$ and no interior zero',
         fontsize=11)
ax2.set_title('B. Interface-location sensitivity of the crossing defect', fontweight='bold')
ax2.set_xlabel(r'interface fraction $\theta$')
ax2.set_ylabel('coefficient magnitude')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.95)
ax2.grid(True, alpha=0.28)
ax2.legend(frameon=False, loc='upper left')

fig.suptitle('Why BDF2 loses smooth-step behavior at an isolated temporal interface',
             fontsize=19, fontweight='bold')

OUTDIR = Path(__file__).resolve().parents[1] / 'figures'
OUTDIR.mkdir(exist_ok=True)
plt.savefig(OUTDIR / 'figure_1_concept.png', dpi=600, bbox_inches='tight')
plt.savefig(OUTDIR / 'figure_1_concept.pdf', bbox_inches='tight')
