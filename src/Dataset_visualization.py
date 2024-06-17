#!/usr/bin/env python
# coding: utf-8
# (c) Charles Le Losq 2021
# see embedded licence file

import numpy as np
import scipy, h5py, matplotlib, torch
import pandas as pd
import matplotlib.pyplot as plt

import imelt
device = torch.device('cpu')

import mpltern
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

###
# Make Figures Functions
###
def figure1(ds, savepath="../figures/datasets/Figure_TernaryData.pdf"):
    plt.figure(figsize=(10,10),dpi=150)

    colors = ['dodgerblue', 'tomato', 'limegreen']
    ax1 = plt.subplot(3,3,1,projection='ternary',ternary_scale=100)
    ax2 = plt.subplot(3,3,2,projection='ternary',ternary_scale=100)
    ax3 = plt.subplot(3,3,3,projection='ternary',ternary_scale=100)
    ax4 = plt.subplot(3,3,4,projection='ternary',ternary_scale=100)
    ax5 = plt.subplot(3,3,5,projection='ternary',ternary_scale=100)
    ax6 = plt.subplot(3,3,6,projection='ternary',ternary_scale=100)
    ax7 = plt.subplot(3,3,7,projection='ternary',ternary_scale=100)
    ax8 = plt.subplot(3,3,8,projection='ternary',ternary_scale=100)
    ax9 = plt.subplot(3,3,9,projection='ternary',ternary_scale=100)

    # The data
    ax1.scatter(ds.x_visco_train.detach().numpy()[:,0], ds.x_visco_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Train data subset")
    ax1.scatter(ds.x_visco_valid.detach().numpy()[:,0], ds.x_visco_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation data subset")
    ax1.scatter(ds.x_visco_test.detach().numpy()[:,0], ds.x_visco_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing data subset")

    ax2.scatter(ds.x_raman_train.detach().numpy()[:,0], ds.x_raman_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_raman_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Ent.")
    ax2.scatter(ds.x_raman_valid.detach().numpy()[:,0], ds.x_raman_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_raman_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Valid.")

    ax3.scatter(ds.x_density_train.detach().numpy()[:,0], ds.x_density_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_density_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Train")
    ax3.scatter(ds.x_density_valid.detach().numpy()[:,0], ds.x_density_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_density_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Valid.")
    ax3.scatter(ds.x_density_test.detach().numpy()[:,0], ds.x_density_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_density_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Test")

    ax4.scatter(ds.x_ri_train.detach().numpy()[:,0], ds.x_ri_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_ri_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Ent.")
    ax4.scatter(ds.x_ri_valid.detach().numpy()[:,0], ds.x_ri_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_ri_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Valid.")
    ax4.scatter(ds.x_ri_test.detach().numpy()[:,0], ds.x_ri_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_ri_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Test")

    ax5.scatter(ds.x_cpl_train.detach().numpy()[:,0], ds.x_cpl_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Training")
    ax5.scatter(ds.x_cpl_valid.detach().numpy()[:,0], ds.x_cpl_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation")
    ax5.scatter(ds.x_cpl_test.detach().numpy()[:,0], ds.x_cpl_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing")

    ax6.scatter(ds.x_abbe_train.detach().numpy()[:,0], ds.x_abbe_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_abbe_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Training")
    ax6.scatter(ds.x_abbe_valid.detach().numpy()[:,0], ds.x_abbe_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_abbe_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation")
    ax6.scatter(ds.x_abbe_test.detach().numpy()[:,0], ds.x_abbe_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_abbe_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing")

    ax7.scatter(ds.x_elastic_train.detach().numpy()[:,0], ds.x_elastic_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_elastic_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Training")
    ax7.scatter(ds.x_elastic_valid.detach().numpy()[:,0], ds.x_elastic_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_elastic_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation")
    ax7.scatter(ds.x_elastic_test.detach().numpy()[:,0], ds.x_elastic_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_elastic_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing")

    ax8.scatter(ds.x_cte_train.detach().numpy()[:,0], ds.x_cte_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_cte_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Training")
    ax8.scatter(ds.x_cte_valid.detach().numpy()[:,0], ds.x_cte_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_cte_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation")
    ax8.scatter(ds.x_cte_test.detach().numpy()[:,0], ds.x_cte_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_cte_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing")

    ax9.scatter(ds.x_liquidus_train.detach().numpy()[:,0], ds.x_liquidus_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_liquidus_train.detach().numpy()[:,1],
                marker='s', s=3, color=colors[0], alpha=1., label="Training")
    ax9.scatter(ds.x_liquidus_valid.detach().numpy()[:,0], ds.x_liquidus_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_liquidus_valid.detach().numpy()[:,1],
                marker='o', s=3, color=colors[1], alpha=1., label="Validation")
    ax9.scatter(ds.x_liquidus_test.detach().numpy()[:,0], ds.x_liquidus_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_liquidus_test.detach().numpy()[:,1],
                marker='d', s=3, color=colors[2], alpha=1., label="Testing")

    # The glass forming domain

    t = [1.0, 0.5, 0.40, 0.48, 0.6, 0.75, 0.95]
    l = [0.0, 0.5, 0.4, 0.26, 0.12, 0.05, 0.0]
    r = [0.0, 0.0, 0.2, 0.26, 0.28, 0.2, 0.05]

    #ax1.fill(t, l, r, "",color="grey", alpha=0.1,label="Domaine de formation des verres")
    #ax2.fill(t, l, r, "",color="grey", alpha=0.1)
    #ax3.fill(t, l, r, "",color="grey", alpha=0.1)
    #ax4.fill(t, l, r, "",color="grey", alpha=0.1)

    ax1.grid(axis='t')
    ax1.grid(axis='l')
    ax1.grid(axis='r')

    ax2.grid(axis='t')
    ax2.grid(axis='l')
    ax2.grid(axis='r')

    ax3.grid(axis='t')
    ax3.grid(axis='l')
    ax3.grid(axis='r')

    ax4.grid(axis='t')
    ax4.grid(axis='l')
    ax4.grid(axis='r')

    ax5.grid(axis='t')
    ax5.grid(axis='l')
    ax5.grid(axis='r')

    ax1.set_tlabel('SiO$_2$')
    ax2.set_tlabel('SiO$_2$')
    ax3.set_tlabel('SiO$_2$')

    ax1.set_llabel('$\sum$M$^{2/x+}_{x}$O')
    ax4.set_llabel('$\sum$M$^{2/x+}_{x}$O')
    ax7.set_llabel('$\sum$M$^{2/x+}_{x}$O')

    ax3.set_rlabel('Al$_2$O$_3$')
    ax6.set_rlabel('Al$_2$O$_3$')
    ax9.set_rlabel('Al$_2$O$_3$')

    ax1.annotate("(a) $D_{viscosity}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax2.annotate("(b) $D_{Raman}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax3.annotate("(c) $D_{density}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax4.annotate("(d) $D_{optical}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax5.annotate("(e) $D_{C_{p}^{liquid}}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax6.annotate("(e) $D_{Abbe}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax7.annotate("(e) $D_{Elastic}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax8.annotate("(e) $D_{CTE}$",xy=(-0.1,0.97),xycoords="axes fraction")
    ax9.annotate("(e) $D_{liquidus}$",xy=(-0.1,0.97),xycoords="axes fraction")

    plt.tight_layout()

    ax3.legend(loc=(1.0,0.7),fontsize=9)
    plt.savefig(savepath)
    plt.close()

def figure2(ds, savepath="../figures/datasets/Figure_TernaryData_forpresentations.pdf"):
    ###
    # For presentations
    ###

    plt.figure(figsize=(8,3.5),dpi=90)

    colors = ['dodgerblue', 'tomato', 'limegreen']

    ax1 = plt.subplot(1,2,1,projection='ternary',ternary_scale=100)
    ax3 = plt.subplot(1,2,2,projection='ternary',ternary_scale=100)

    # The data
    ax1.scatter(ds.x_visco_train.detach().numpy()[:,0], ds.x_visco_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_train.detach().numpy()[:,1],
                marker='s', s=5, color=colors[0], alpha=1., label="Train data subset")
    ax1.scatter(ds.x_visco_valid.detach().numpy()[:,0], ds.x_visco_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_valid.detach().numpy()[:,1],
                marker='o', s=5, color=colors[1], alpha=1., label="Validation data subset")
    ax1.scatter(ds.x_visco_test.detach().numpy()[:,0], ds.x_visco_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_visco_test.detach().numpy()[:,1],
                marker='d', s=5, color=colors[2], alpha=1., label="Testing data subset")

    ax3.scatter(ds.x_cpl_train.detach().numpy()[:,0], ds.x_cpl_train.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_train.detach().numpy()[:,1],
                marker='s', s=5, color=colors[0], alpha=1., label="Ent.")
    ax3.scatter(ds.x_cpl_valid.detach().numpy()[:,0], ds.x_cpl_valid.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_valid.detach().numpy()[:,1],
                marker='o', s=5, color=colors[1], alpha=1., label="Valid.")
    ax3.scatter(ds.x_cpl_test.detach().numpy()[:,0], ds.x_cpl_test.detach().numpy()[:,2:6].sum(axis=1), ds.x_cpl_test.detach().numpy()[:,1],
                marker='d', s=5, color=colors[2], alpha=1., label="Test")

    # The glass forming domain

    t = [1.0, 0.5, 0.40, 0.48, 0.6, 0.75, 0.95]
    l = [0.0, 0.5, 0.4, 0.26, 0.12, 0.05, 0.0]
    r = [0.0, 0.0, 0.2, 0.26, 0.28, 0.2, 0.05]

    #ax1.fill(t, l, r, "",color="grey", alpha=0.1,label="Domaine de formation des verres")
    #ax2.fill(t, l, r, "",color="grey", alpha=0.1)
    #ax3.fill(t, l, r, "",color="grey", alpha=0.1)
    #ax4.fill(t, l, r, "",color="grey", alpha=0.1)

    ax1.grid(axis='t')
    ax1.grid(axis='l')
    ax1.grid(axis='r')

    ax3.grid(axis='t')
    ax3.grid(axis='l')
    ax3.grid(axis='r')

    ax1.set_tlabel('SiO$_2$')
    ax1.set_llabel('Na$_2$O+K$_2$O+CaO+MgO')
    ax1.set_rlabel('Al$_2$O$_3$')

    ax3.set_tlabel('SiO$_2$')
    ax3.set_llabel('Na$_2$O+K$_2$O+CaO+MgO')
    ax3.set_rlabel('Al$_2$O$_3$')

    ax1.annotate("(a) Viscosity",xy=(-0.3,0.9),xycoords="axes fraction")
    ax3.annotate("(b) Heat Capacity",xy=(-0.3,0.9),xycoords="axes fraction")

    plt.tight_layout()

    ax1.legend(loc=(.4,-0.5),fontsize=9)
    plt.savefig(savepath,bbox_inches='tight')
    plt.close()

def figure3(ds, savepath="../figures/datasets/Figure_repartition.pdf"):
    
    ###
    # # Quantity of each chemical component in the dataset
    ###

    # viscosity
    x_visco = torch.cat((ds.x_visco_train, ds.x_visco_valid, ds.x_visco_test)).unique(dim=0)
    nb_visco = []
    for i in range(6):
        nb_visco.append(len(np.where(x_visco[:,i]!=0)[0]))

    #densité
    x_density = torch.cat((ds.x_density_train, ds.x_density_valid, ds.x_density_test)).unique(dim=0)
    nb_density = []
    for i in range(6):
        nb_density.append(len(np.where(x_density[:,i]!=0)[0]))

    # #refractive index
    # x_ri = torch.cat((ds.x_ri_train, ds.x_ri_valid, ds.x_ri_test)).unique(dim=0)
    # #print(len(x_ri))
    # nb_sio2 = len(np.where(x_ri[:,0]!=0)[0])
    # nb_al2o3 = len(np.where(x_ri[:,1]!=0)[0])
    # nb_na2o = len(np.where(x_ri[:,2]!=0)[0])
    # nb_k2o = len(np.where(x_ri[:,3]!=0)[0])
    # nb_mgo = len(np.where(x_ri[:,4]!=0)[0])
    # nb_cao = len(np.where(x_ri[:,5]!=0)[0])
    # nb_ri = [nb_sio2, nb_al2o3, nb_na2o, nb_k2o, nb_mgo, nb_cao]
    # #print(nb_ri)

    # #raman
    # x_raman = torch.cat((ds.x_raman_train, ds.x_raman_valid)).unique(dim=0)
    # #print(len(x_raman))
    # nb_sio2 = len(np.where(x_raman[:,0]!=0)[0])
    # nb_al2o3 = len(np.where(x_raman[:,1]!=0)[0])
    # nb_na2o = len(np.where(x_raman[:,2]!=0)[0])
    # nb_k2o = len(np.where(x_raman[:,3]!=0)[0])
    # nb_mgo = len(np.where(x_raman[:,4]!=0)[0])
    # nb_cao = len(np.where(x_raman[:,5]!=0)[0])
    # nb_raman = [nb_sio2, nb_al2o3, nb_na2o, nb_k2o, nb_mgo, nb_cao]
    # #print(nb_raman)

    fig = plt.figure(figsize=(3.22,6), dpi=150)

    colors=['lightgreen', 'mediumaquamarine', 'turquoise', 'deepskyblue', 'dodgerblue', 'royalblue']

    ax = plt.subplot(211)
    plt.bar(range(0,6),nb_visco, tick_label=['SiO$_2$', 'Al$_2$O$_3$', 'Na$_2$O', 'K$_2$O', 'MgO', 'CaO'],color=colors)
    plt.annotate("(a) Viscosity",xy=(0.95,0.9),xycoords="axes fraction",ha="right")
    for count,i in enumerate(nb_visco):
        plt.annotate(str(i),xy=(count,i-50),xycoords="data",color='w', ha='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax = plt.subplot(212)
    # plt.bar(range(0,6),nb_raman, tick_label=['SiO$_2$', 'Al$_2$O$_3$', 'Na$_2$O', 'K$_2$O', 'MgO', 'CaO'],color=colors)
    # plt.annotate("(b) Raman \n spectra",xy=(0.95,0.82),xycoords="axes fraction",ha="right")
    # plt.annotate(str(nb_raman[0]),xy=(0.055,0.86),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_raman[1]),xy=(0.21,0.765),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_raman[2]),xy=(0.385,0.28),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_raman[3]),xy=(0.54,0.2),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_raman[4]),xy=(0.695,0.07),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_raman[5]),xy=(0.855,0.27),xycoords="axes fraction",color='w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    ax = plt.subplot(212)
    plt.bar(range(0,6),nb_density, tick_label=['SiO$_2$', 'Al$_2$O$_3$', 'Na$_2$O', 'K$_2$O', 'MgO', 'CaO'],color=colors)
    plt.annotate("(b) Density",xy=(0.95,0.9),xycoords="axes fraction",ha="right")
    for count,i in enumerate(nb_density):
        plt.annotate(str(i),xy=(count,i-50),xycoords="data",color='w', ha='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax = plt.subplot(224)
    # plt.bar(range(0,6),nb_ri, tick_label=['SiO$_2$', 'Al$_2$O$_3$', 'Na$_2$O', 'K$_2$O', 'MgO', 'CaO'],color=colors)
    # plt.annotate("(d) Optical\nrefractive\nindex",xy=(0.95,0.82),xycoords="axes fraction",ha="right")
    # plt.annotate(str(nb_ri[0]),xy=(0.055,0.86),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_ri[1]),xy=(0.21,0.12),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_ri[2]),xy=(0.365,0.57),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_ri[3]),xy=(0.52,0.175),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_ri[4]),xy=(0.695,0.05),xycoords="axes fraction",color='w')
    # plt.annotate(str(nb_ri[5]),xy=(0.835,0.39),xycoords="axes fraction",color='w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    fig.text(0.0, 0.5, "Number of compositions including the element", va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

if __name__=="__main__":
    ###
    # Load dataset
    ###
    ds = imelt.data_loader()
    ds.print_data()
    #figure1(ds)
    #figure2(ds)
    figure3(ds)
