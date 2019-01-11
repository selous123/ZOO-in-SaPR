import matplotlib.pyplot as plt
def plot_save_fig(data,x,y,xlabel,title,xticks,yticks,istitle,ax,legend_fontsize = 20,ncol=1,legend_label=False):



#     plt.figure(figsize=(6, 4.5))
    #ax = plt.gca()
    ax.spines['top'].set_visible(False)
#     #ax.spines['bottom'].set_visible(False)
#     #ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend_label:
        ax.plot(x,data[y[0]],linestyle="-", linewidth=1,marker="o",markersize=10,color="red",label="ZOO")
        ax.plot(x,data[y[1]],linestyle="-.", linewidth=1,marker="v",markersize=10,color="black",label="LR")
    #     #plt.plot(x,mean_simo_score[2],linestyle="--", linewidth=1)
    #     #plt.plot(x,mean_simo_score[3]+0.3,linestyle=":", linewidth=1)
    #     #plt.plot(x,mean_simo_score[4],linestyle="-.", linewidth=1)
        ax.plot(x,data[y[2]],linestyle="-", linewidth=1,marker="*",markersize=10,color="blue",label="HR")
    else:
        ax.plot(x,data[y[0]],linestyle="-", linewidth=1,marker="o",markersize=10,color="red")
        ax.plot(x,data[y[1]],linestyle="-.", linewidth=1,marker="v",markersize=10,color="black")
#     #plt.plot(x,mean_simo_score[2],linestyle="--", linewidth=1)
#     #plt.plot(x,mean_simo_score[3]+0.3,linestyle=":", linewidth=1)
#     #plt.plot(x,mean_simo_score[4],linestyle="-.", linewidth=1)
        ax.plot(x,data[y[2]],linestyle="-", linewidth=1,marker="*",markersize=10,color="blue")
    ax.set_ylabel(r"$||w-w^*||_2$")
    ax.set_xlabel(xlabel)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

#     #plt.xticks([])
#     #plt.yticks([])

#     #plt.title(r"Fix n$=1500,\sigma=1,\alpha=0.5$")

#     #size = fig.get_size_inches()*fig.dpi # get fig size in pixels

    ax.tick_params(direction='in')
    #ax.legend(fontsize=legend_fontsize,ncol=ncol,fancybox=True, framealpha=0.5)
#     plt.tight_layout()
#     if istitle:
    ax.set_title(title, y=-0.45,fontweight='bold')
#     plt.savefig(name, format='eps', bbox_inches='tight', pad_inches = 0)
    #plt.show()
