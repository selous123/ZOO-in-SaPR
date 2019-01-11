import numpy as np
from utils import plot_save_fig
def prepross_data(score):
    score = np.array(score)
    mean_simo_score = score[:,:,0]/10
    return mean_simo_score


import matplotlib
import matplotlib.pyplot as plt
#set rcParams
matplotlib.rc('axes',titlesize=25)
matplotlib.rc('axes', labelsize=25)
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)
fig,axs = plt.subplots(1,4,figsize=(24,4.5),constrained_layout=True)
axs = axs.flatten()


m_score = [
[
0.9221 ,1.5236 ,1.5236 ,1.5231 ,1.6176 ,1.7768] ,
 [1.1912 ,1.8995 ,1.8995 ,1.9043 ,2.0698 ,2.2750] ,
  [1.7268 ,2.5091 ,2.5091 ,2.5137 ,2.7705 ,3.0142] ,
   [2.0689 ,2.9073 ,2.9073 ,2.9074 ,3.2390 ,3.4855] ,
    [2.7642 ,3.7370 ,3.7370 ,3.7447 ,4.1530 ,4.2868 ]
]
data = np.array(m_score)
data = data.T
x = [600,700,800,900,1000]
y = [0,1,5]
title=r'(b) Fix n=1500,$\sigma$=1,$\alpha$=0.5'
xticks=[600,800,1000]
yticks = [0.5,2.5,4.5]
#istitle=False
#name = "sample_notitle.png"
istitle=True

plot_save_fig(data,x,y,xlabel="m",title=title,xticks=xticks,yticks=yticks,istitle=istitle,ax=axs[1],legend_label=True)
#legend = plt.figlegend(fontsize=20,ncol=3,loc=9,bbox_to_anchor=(0.5, 1.1))


n_score = [
[1.1710, 1.7915, 1.7915, 1.7995, 1.9600, 2.2958 ],
[0.9303, 1.4907, 1.4906, 1.4916, 1.5911, 1.6487 ],
[0.7137, 1.2802, 1.2802, 1.2802, 1.3550, 1.2467 ],
[0.6480 ,1.2344, 1.2344, 1.2367, 1.3146, 1.1650],
[0.5735 ,1.0996 ,1.0996 ,1.1020 ,1.1852 ,1.0197 ]
]
data = np.array(n_score)
data = data.T
x = [1000,1200,1400,1600,1800]
y = [0,1,5]
title=r'(a) Fix m=500,$\sigma$=1,$\alpha$=0.5'
xticks=[1000,1400,1800]
yticks = [0.5,1.5,2.5]
#istitle=False
#name = "sample_notitle.png"
istitle=True
plot_save_fig(data,x,y,xlabel="n",title=title,xticks=xticks,yticks=yticks,istitle=istitle,ax=axs[0]) 


#sigma
sigma_score = [
[0.5915, 0.8958, 0.8958, 0.8977, 0.9585, 0.8972 ],
[0.7504, 1.0192, 1.0192, 1.0208, 1.0815, 1.0504 ],
[0.9182, 1.0903, 1.0903, 1.0931 ,1.1181, 1.1645 ],
[1.1122, 1.2704, 1.2704, 1.2721, 1.3256, 1.3434 ],
[1.2402, 1.3163, 1.3163, 1.3163, 1.3517, 1.4263 ]
]
data = np.array(sigma_score)
data = data.T
x = [2,3,4,5,6]
y = [0,1,5]
title=r'(c) Fix m=400,n=1800,$\alpha$=0.5'
xticks=[2,4,6]
yticks = [0.50,1.00,1.50]
#istitle=False
#name = "sample_notitle.png"
istitle=True
plot_save_fig(data,x,y,xlabel=r'$\sigma$',title=title,xticks=xticks,yticks=yticks,istitle=istitle,ax=axs[2],legend_fontsize = 14,ncol=3) 

#alpha
alpha_score = [
[0.5791 ,0.9992 ,0.9992 ,1.0008 ,1.0577 ,1.0101],
 [0.6589 ,1.0278 ,1.0278 ,1.0286 ,1.0807 ,1.1237 ],
  [0.7654 ,1.1200 ,1.1200 ,1.1209 ,1.1724 ,1.2507],
   [0.9116 ,1.1743 ,1.1743 ,1.1775 ,1.1972 ,1.3781 ],
    [1.0204 ,1.2025 ,1.2025 ,1.2028 ,1.2440 ,1.4678 ]
]

data = np.array(alpha_score)
data = data.T

#data = prepross_data(alpha_score)
x = [0.6,0.7,0.8,0.9,1.0]
y = [0,1,5]
title=r'(d) Fix m=400,n=1800,$\sigma$=1'
#yticks = [0.25,0.35,0.45,0.55,0.65,0.75]
xticks = [0.6,0.8,1.0]
yticks = [0.5,1.0,1.5]
#istitle=False
#name = "sample_notitle.png"
istitle=True
plot_save_fig(data,x,y,xlabel=r'$\alpha$',title=title,xticks=xticks,yticks=yticks,istitle=istitle,ax=axs[3]) 


name = "samples.pdf"
fig.tight_layout()
#fig.subplots_adjust(top=0.75)
plt.subplots_adjust(wspace=0.25, hspace=0)
#axs[0].autoscale_view()
#fig.artists.append(legend)
#legend = plt.figlegend(fontsize=20,ncol=3,loc=9,bbox_to_anchor=(0.5, 1.5))
legend = fig.legend(fontsize=25,ncol=3,loc=9,bbox_to_anchor=(0.5, 1.2))
fig.savefig(name,format='pdf', bbox_inches='tight', pad_inches = 0)
print "save in samples.pdf"
#plt.show()
