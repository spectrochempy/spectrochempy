# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(3.6,5), dpi=100, frameon=False)
ax = fig.add_subplot(111)
agir = plt.imread('agir_setup.png')
ax.imshow(agir)
ax.axis('off')
    
fs = 10

ax.annotate('Balance', xy=(3200, 1000),  xycoords='data',fontsize=fs,
                xytext=(-55, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='red',
                                connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round", fc="0.95"),
                )
ax.annotate('IR cell reactor', xy=(3000, 4400),  xycoords='data',fontsize=fs,
                xytext=(-105, -42), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='red',
                                connectionstyle="arc3,rad=-.2"),
                bbox=dict(boxstyle="round", fc="0.95"),
                )

ax.annotate('Gas flow\n   in/out', xy=(3000, 3200),  xycoords='data',fontsize=fs,
                xytext=(-40, 25), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='red',
                                connectionstyle="arc3,rad=-.2"),
                bbox=dict(boxstyle="round", fc="0.95"),
                )

ax.annotate('Sample\nholder', xy=(1050, 1000),  xycoords='data',fontsize=fs,
                xytext=(20, -25), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='red',
                                connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round", fc="0.95"),
                )

ax.annotate('Pellet', xy=(1250, 4250),  xycoords='data', fontsize=fs,
                xytext=(3, 18), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='red',
                                connectionstyle="arc3,rad=-.2"),
                bbox=dict(boxstyle="round", fc="0.95"),
                )

plt.subplots_adjust(top=1., bottom=0.0, left=0.0, right=1.0)#, wspace=0.5)
ax.set_xlim(500,5400)
extension = '.jpg'
fig.savefig('annotated_fig_agir_setup%s'%extension,figsize=(3.6,5), dpi=100)


# %%
