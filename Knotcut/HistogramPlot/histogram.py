import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec


def histogram(N, y_r, y_g, y_b):
    color_syn = ('b','g','r')
    color_name = ('blue', 'green', 'red')
    
    x = range(0,256,1)
    
    plt.figure(facecolor='white')
    
    gs = gridspec.GridSpec(3, 1)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[10, 1], hspace=0)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[10, 1], hspace=0)
    gs3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], height_ratios=[10, 1], hspace=0)   
    
    ax_plot_1 = plt.subplot(gs1[0])
    ax_bar_1  = plt.subplot(gs1[1])
    plt.setp(ax_plot_1.get_yticklabels(), visible=False)
    plt.setp(ax_plot_1.get_xticklabels(), visible=False)
    
    ax_plot_2 = plt.subplot(gs2[0], sharey=ax_plot_1)
    ax_bar_2  = plt.subplot(gs2[1])
    plt.setp(ax_plot_2.get_yticklabels(), visible=False)
    plt.setp(ax_plot_2.get_xticklabels(), visible=False)
    
    ax_plot_3 = plt.subplot(gs3[0], sharey=ax_plot_1)
    ax_bar_3  = plt.subplot(gs3[1])
    plt.setp(ax_plot_3.get_yticklabels(), visible=False)
    plt.setp(ax_plot_3.get_xticklabels(), visible=False)
       
    ax_plot_1.plot(y_r, color = 'r')
    ax_plot_1.fill_between(x, 0, y_r, facecolor='red', alpha=0.5)       
    ax_plot_1.set_xlim([0,255])
    ax_plot_1.set_ylim([0,1])
    
    ax_plot_2.plot(y_g, color = 'g')
    ax_plot_2.fill_between(x, 0, y_g, facecolor='green', alpha=0.5)       
    ax_plot_2.set_xlim([0,255])
    ax_plot_2.set_ylim([0,1])
    
    ax_plot_3.plot(y_b, color = 'b')
    ax_plot_3.fill_between(x, 0, y_b, facecolor='blue', alpha=0.5)       
    ax_plot_3.set_xlim([0,255])
    ax_plot_3.set_ylim([0,1])
    
    cm_r = mpl.cm.get_cmap("Reds_r")
    cm_g = mpl.cm.get_cmap("Greens_r")
    cm_b = mpl.cm.get_cmap("Blues_r")
    cmap = (cm_r, cm_g, cm_b)
    norm = mpl.colors.Normalize(vmin=0, vmax=255)
    
    cb0 = mpl.colorbar.ColorbarBase(ax_bar_1, cmap=cmap[0], norm=norm, orientation='horizontal')
    cb0.set_label('Red')
    
    cb1 = mpl.colorbar.ColorbarBase(ax_bar_2, cmap=cmap[1], norm=norm, orientation='horizontal')
    cb1.set_label('Green')
    
    cb2 = mpl.colorbar.ColorbarBase(ax_bar_3, cmap=cmap[2], norm=norm, orientation='horizontal')
    cb2.set_label('Blue')
    
    plt.tight_layout() 
    plt.show()

    return N


def main():
    img = cv2.imread('D:\\Slike\\Samples\\grca_RGB_000298.bmp')
    N = 256
    
    hist_r = cv2.calcHist([img],[0],None,[N],[0,N])
    hist_g = cv2.calcHist([img],[1],None,[N],[0,N])
    hist_b = cv2.calcHist([img],[2],None,[N],[0,N])

    y_r = [item for sublist in hist_r.tolist() for item in sublist]
    y_g = [item for sublist in hist_g.tolist() for item in sublist]
    y_b = [item for sublist in hist_b.tolist() for item in sublist]
    
    histogram(N, y_r, y_g, y_b)

if __name__ == "__main__":
    main()