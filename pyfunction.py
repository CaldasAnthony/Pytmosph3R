import math as math
from pyconstant import *

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import sys, time, os

########################################################################################################################
########################################################################################################################

"""
    PYFUNCTION

    Cette bibliotheque rassemble l'ensemble des "petites" fonctions necessaires au bon fonctionnement des cinq autres
    bibliotheques pygcmtreat, pydataread, pykcorr, pyremind et pytransfert. Ces fontcions interviennent dans les calculs
    systematiques, le traitement de donnee comme le suivi de la progression des routines (ProgressBar). Certaines de ces
    fonctions ne sont plus utilisees dans l'absolue mais pourront l'etre a l'avenir, nous avons donc fait le choix de
    les conserver.

    Version : 6.2

    Date de derniere modification : 06.07.2016

    >> Insertion des fonctions de modulation de la densite et de la calibration

    Date de derniere modification : 12.12.2016

    >> Fonctions d'interpolation

    Date de derniere modification : 06.06.2017

    >> Fonctions d'enregistrement des fichiers
    >> Fonctions d'accompagnement des fonctions de pygcmtreat (rendre la bibliotheque plus lisible et pouvoir ainsi
    la mettre a jour plus aisement)

"""

########################################################################################################################
########################################################################################################################


def resolution_convertator(I,wavenumber_ref,bande_ref,R_s,Rp,r_step,extra,trans,Square=True,Kcorr=False) :

    from pytransfert import atmospectre

    R_mean_graph = np.array([])
    bande_graph = np.array([])

    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar = atmospectre(I,wavenumber_ref,R_s,Rp,r_step,extra,trans)

    if Kcorr == True :

        for i in range(bande_ref.size-1) :

            wh, = np.where((wavenumber_ref >= bande_ref[i])*(wavenumber_ref <= bande_ref[i+1]))

            R_mean = np.nansum(np.sqrt(R_eff[wh[0]-1]**2))*(wavenumber_ref[wh[0]]-bande_ref[i])/(wavenumber_ref[wh[0]]-wavenumber_ref[wh[0]-1])

            R_mean += np.nansum(np.sqrt(R_eff[wh]**2))/(float(wh.size))
            R_mean += np.nansum(np.sqrt(R_eff[wh[wh.size-1]+1]**2))*(bande_ref[i+1]-wavenumber_ref[wh[wh.size-1]])/(wavenumber_ref[wh[wh.size-1]+1]-wavenumber_ref[wh[wh.size-1]])

            R_mean_graph = np.append(R_mean_graph,R_mean)
            R_mean_graph = np.append(R_mean_graph,R_mean)

            bande_graph = np.append(bande_graph,bande_ref[i])
            bande_graph = np.append(bande_graph,bande_ref[i+1])

    else :

        for i in range(bande_ref.size-1) :

            wh, = np.where((wavenumber_ref >= bande_ref[i])*(wavenumber_ref <= bande_ref[i+1]))

            if Square == True :
                R_mean = np.sqrt(np.nansum(R_eff[wh]**2)/(float(wh.size)))
            else :
                R_mean = np.nansum(R_eff[wh])/(float(wh.size))

            R_mean_graph = np.append(R_mean_graph,R_mean)
            R_mean_graph = np.append(R_mean_graph,R_mean)

            bande_graph = np.append(bande_graph,bande_ref[i])
            bande_graph = np.append(bande_graph,bande_ref[i+1])

    return R_mean_graph,bande_graph


########################################################################################################################


def stud_type(r_eff,single,Continuum=False,Isolated=False,Scattering=False,Clouds=False) :

    stu = np.array([])

    if Isolated == True :

        if Continuum == True :
            stu = np.append(stu,np.array(["cont"]))

        if Scattering == True :
            stu = np.append(stu,np.array(["sca"]))

        if Clouds == True :
            if single == "no" :
                stu = np.append(stu,np.array(["cloud"]))
            else :
                stu = np.append(stu,np.array([single]))

    else :

        if Continuum == True :
            stu = np.append(stu,np.array(["cont"]))

        if Scattering == True :
            stu = np.append(stu,np.array(["sca"]))

        if Clouds == True :
            if single == "no" :
                stu = np.append(stu,np.array(["cloud"]))
            else :
                stu = np.append(stu,np.array([single]))

        stu = np.append(stu,np.array(["tot"]))

    link = stu.size

    if link == 1 :

        if Isolated == True :
            if Clouds == True :
                stud = "%.2f_%s"%(r_eff*10**6,stu[0])
            else :
                stud = "%s"%(stu[0])
        else :
            stud = "%s"%("nude")

    if link == 2 :

        if Clouds == True :
            stud = "%.2f_%s%s"%(r_eff*10**6,stu[0],stu[1])
        else :
            stud = "%s%s"%(stu[0],stu[1])

    if link == 3 :

        if Clouds == True :
            stud = "%.2f_%s%s%s"%(r_eff*10**6,stu[0],stu[1],stu[2])
        else :
            stud = "%s%s%s"%(stu[0],stu[1],stu[2])

    if link == 4 :

        if Clouds == True :
            stud = "%.2f_tot"%(r_eff*10**6)
        else :
            stud = "tot"

    return stud


########################################################################################################################


def saving(dimension,type,special,save_adress,version,name,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,phi_rot,r_eff,\
           domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,D1) :

    s_n = special

    if Discreet == True :

        if s_n == '' :
            s_n += 'dis'
        else :
            s_n += '_dis'

    if Integration == True :

        if s_n == '' :
            s_n += 'int'
        else :
            s_n += '_int'

    if Optimal == True :

        if s_n == '' :
            s_n += 'opt'
        else :
            s_n += '_opt'

    if Module == True :

        if s_n == '' :
            if np.str(type[0]) != 'specific' :
                s_n += '%s'%(type[0])
            else :
                s_n += 'spe%.2f'%(type[1])
        else :
            if np.str(type[0]) != 'specific' :
                s_n += '_%s'%(type[0])
            else :
                s_n += '_spe%.2f'%(type[1])

    if rupt_alt != 0 :
        h_range = '%i:%i'%(rupt_alt/r_step,lim_alt/r_step)
    else :
        h_range = '%i'%(lim_alt/r_step)
    if dimension == '1D' :
        D = 1
    if dimension == '3D' :
        D = 3

    if Kcorr == True :
        if D1 == False :
            s_m = '%sI_%s_%.1f_%s_%i_%i%i_%i_%i%i_%s_%i_%.2f_%.2f_%s_%s'\
                %(save_adress,s_n,version,name,D,reso_long,reso_lat,t,dim_bande,dim_gauss,h_range,r_step,phi_rot,r_eff*10**6,\
                  stud,domain)
        else :
            s_m = '%sI_%s_%.1f_%s_%i_%i_%i_%i%i_%i_%i%i_%s_%i_%.2f_%.2f_%s_%s'\
                %(save_adress,s_n,version,name,D,long,lat,reso_long,reso_lat,t,dim_bande,dim_gauss,h_range,r_step,phi_rot,r_eff*10**6,\
                  stud,domain)
    else :
        if D1 == False :
            s_m = "%sI_%s_%.1f_%s_%i_%i%i_%i_%i_%s_%i_%.2f_%.2f_%s_%s"\
                %(save_adress,s_n,version,name,D,reso_long,reso_lat,t,dim_bande,h_range,r_step,phi_rot,r_eff*10**6,\
                    stud,domain)
        else :
            s_m = "%sI_%s_%.1f_%s_%i_%i_%i_%i%i_%i_%i_%s_%i_%.2f_%.2f_%s_%s"\
                %(save_adress,s_n,version,name,D,long,lat,reso_long,reso_lat,t,dim_bande,h_range,r_step,phi_rot,r_eff*10**6,\
                  stud,domain)

    return s_m


########################################################################################################################


def index_active(n_species,n_species_cross,n_species_active):

    ind_active = np.zeros((n_species_active.size),dtype='int')
    ind_cross = np.zeros((n_species_active.size),dtype='int')
    for nspa in range(n_species_active.size) :
        wh, = np.where(n_species == n_species_active[nspa])
        ind_active[nspa] = int(wh[0])
        wh, = np.where(n_species_cross == n_species_active[nspa])
        ind_cross[nspa] = int(wh[0])

    return ind_cross, ind_active


########################################################################################################################


def ratio(n_species,x_ratio_species_active,IsoComp=False) :

    M_species = np.zeros(n_species.size)
    for n_sp in range(n_species.size) :
        wh, = np.where(M_n_mole == n_species[n_sp])
        if wh.size == 0 :
            wh, = np.where(M_n_atom == n_species[n_sp])
            M_species[n_sp] = M_atom[wh[0]]
        M_species[n_sp] = M_mole[wh[0]]

    if IsoComp == True :
        x_ratio_species = np.zeros((n_species.size))
        x_ratio_species[0] = (1 - np.nansum(x_ratio_species_active))/(1. + ratio_HeH2)
        x_ratio_species[1] = ratio_HeH2*x_ratio_species[0]
        x_ratio_species[2:] = x_ratio_species_active[:]
        M = np.nansum(x_ratio_species*M_species)
    else :
        x_ratio_species = np.array([])
        M = 0.

    return M_species,M,x_ratio_species


########################################################################################################################


def diag(data_base) :

    file = Dataset("%s.nc"%(data_base))
    variables = file.variables
    controle = variables["controle"][:]
    Rp = controle[4]
    g = controle[6]
    reso_long = controle[0]
    reso_lat = controle[1]

    return Rp,g,reso_long,reso_lat


########################################################################################################################


def sort_set_param(P,T,Q,gen,compo,Marker=False,Clouds=False) :

    sh = np.shape(P)
    ind = 0
    P_rmd = np.zeros(sh[0]*sh[1]*sh[2])
    T_rmd = np.zeros(sh[0]*sh[1]*sh[2])

    bar = ProgressBar(sh[0],'Reduction of the parameters')

    if Marker == True :
        Q_rmd = np.zeros(sh[0]*sh[1]*sh[2])
    else :
        Q_r = 0

    if Clouds == True :
        sh1 = np.shape(gen)
        gen_rmd = np.zeros((sh1[0],sh[0]*sh[1]*sh[2]))
    else :
        gen_r = 0

    sh2 = np.shape(compo)
    compo_rmd = np.zeros((sh2[0],sh[0]*sh[1]*sh[2]))

    for i in range(sh[0]) :

        for j in range(sh[1]) :

            wh, = np.where((P[i,j,:] != 0))

            P_rmd[ind:ind+wh.size] = P[i,j,wh]
            T_rmd[ind:ind+wh.size] = T[i,j,wh]

            if Marker == True :
                Q_rmd[ind:ind+wh.size] = Q[i,j,wh]

            if Clouds == True :
                gen_rmd[:,ind:ind+wh.size] = gen[:,i,j,wh]

            compo_rmd[:,ind:ind+wh.size] = compo[:,i,j,wh]

            ind += wh.size

        bar.animate(i + 1)

    del P,T,Q,compo,gen

    wh, = np.where(P_rmd != 0)
    P_rmd = P_rmd[wh]

    P_s = np.sort(P_rmd)
    indices = np.argsort(P_rmd)

    T_rmd = T_rmd[wh]
    T_s = T_rmd[indices]

    compo_rmd = compo_rmd[:,wh]
    compo_s = compo_rmd[:,indices]
    del compo_rmd

    if Marker == True :
        Q_rmd = Q_rmd[wh]
        Q_s = Q_rmd[indices]
        del Q_rmd

    if Clouds == True :
        gen_rmd = gen_rmd[:,wh]
        gen_s = gen_rmd[:,indices]
        del gen_rmd

    del P_rmd,T_rmd

    bar = ProgressBar(wh.size, 'Sort and set of the parameters')

    list = []

    for ind in xrange(1,wh.size) :

        if Marker == True :
            if P_s[ind] == P_s[ind - 1] and T_s[ind] == T_s[ind - 1] and Q_s[ind] == Q_s[ind - 1]:
                list.append(ind)
        else :
            if P_s[ind] == P_s[ind - 1] and T_s[ind] == T_s[ind - 1] :
                list.append(ind)

        if ind%100000 == 0 or ind == wh.size - 1 :
            bar.animate(ind +1)

    P_r = np.delete(P_s,list)
    T_r = np.delete(T_s,list)
    compo_r = np.delete(compo_s,list,axis=1)

    if Marker == True :
        Q_r = np.delete(Q_s,list)

    if Clouds == True :
        gen_r = np.delete(gen_s,list,axis=1)

    return P_r, T_r, Q_r, gen_r, compo_r, wh, indices, list


########################################################################################################################


def module_density(P_ref,T_ref,z_ref,Rp,g0,M_ref,r_step,type,Middle) :

    size = P_ref.size

    if np.str(type[0]) == 'mean' :

        P_int = np.zeros((101,size))
        P_mean = np.zeros(size)

        for i_P in range(101) :

            z_step = - r_step/2. + i_P*r_step/100.
            z_stud = z_ref + z_step
            P_int[i_P,:] = P_ref*np.exp(-M_ref*g0/(R_gp*T_ref)*z_step/((1+z_ref/Rp)*(1+z_stud/Rp)))

        for i_s in range(size) :

            P_mean[i_s] = np.mean(P_int[:,i_s])

        return P_mean

    if np.str(type[0]) == 'top' :

        z_step = r_step/2.
        z_stud = z_ref + r_step/2.
        P_top = P_ref*np.exp(-M_ref*g0/(R_gp*T_ref)*z_step/((1+z_ref/Rp)*(1+z_stud/Rp)))

        return P_top

    if np.str(type[0]) == 'bottom' :

        z_step = -r_step/2.
        z_stud = z_ref - r_step/2.
        P_bot = P_ref*np.exp(-M_ref*g0/(R_gp*T_ref)*z_step/((1+z_ref/Rp)*(1+z_stud/Rp)))

        return P_bot

    if np.str(type[0]) == 'specific' :

        z_step = -r_step/2. + np.float(type[1])*r_step
        z_stud = z_ref - r_step/2. + np.float(type[1])*r_step
        P_spe = P_ref*np.exp(-M_ref*g0/(R_gp*T_ref)*z_step/((1+z_ref/Rp)*(1+z_stud/Rp)))

        return P_spe


########################################################################################################################


def calibration(R,R_eff,spec,min,max,Rs) :

    R_spec = np.sqrt(spec[1])*Rs
    delta = R_eff[min:max] - R_spec
    corr = np.interp(R,R_spec,delta)

    return R - corr


########################################################################################################################


def maker_contribution_HR(adress,version,name,dim_bande,lim_alt,reso_step,reso_theta,reso_alt,reso_long,reso_lat,phi_rot,reff,number,single,\
                          Isolated=False,Continuum=False,Scattering=False,Clouds=False,Remove=False) :

    reso_cut = dim_bande/number
    type = stud_type(single,Continuum,Isolated,Scattering,Clouds)
    I = np.zeros((dim_bande,reso_alt,reso_theta),dtype=np.float64)

    for i in range(number) :

        step = reso_cut*(i+1)
        I1 = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR%i.npy'\
                        %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type,step))
        I[i*reso_cut:step] = I1

        if Remove == True :
            os.remove('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR%i.npy'\
                      %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type,step))

    np.save('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR.npy'\
            %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type),I)


########################################################################################################################


def mixing_contribution_HR(adress,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,c_name,\
                           Isolated,Continuum,Scattering,Clouds,single) :

    stud = stud_type(single,Continuum,Isolated,Scattering,Clouds)
    t_corr = 0
    t_cont = 0
    t_sca = 0
    t_cloud = 0

    if Isolated == False :

        I_corr = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_nude_HR.npy'\
                         %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_corr = np.log(I_corr)

    if Continuum == True :

        I_cont = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cont_HR.npy'\
                         %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_cont = np.log(I_cont)

    if Scattering == True :

        I_sca = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_sca_HR.npy'\
                        %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_sca = np.log(I_sca)

    if Clouds == True :

        if single == 'no' :
            I_cloud = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cloud_HR.npy'\
                              %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        else :
            I_cloud = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cloud_%s_HR.npy'\
                              %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,c_name[single]))
        t_cloud = np.log(I_cloud)

    tau = t_corr + t_cont + t_sca + t_cloud

    I = np.exp(tau)

    np.save('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR.npy'\
            %(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,stud),I)


########################################################################################################################
########################################################################################################################

"""
    ATMOSPHERE_PLOT

    Cette fonction simple permet de representer dans un espace cartesien les matrices de transmittance des atmospheres
    et eventuellement l'emittance de l'exoplanete. Le rayon qui est alors propose n'est pas realiste de maniere a bien
    visualiser les structures atmospheriques, d'autant plus que le temps de calcul peut devenir enorme pour des
    dimensions plus realistes.

"""

########################################################################################################################
########################################################################################################################


def atmosphere_plot(I_tot,h,param,factor,r_step,theta_number,wl,bande_sample,name,Rp,R_s,extra,trans,Kcorr,Middle) :

    theta_step = 2*np.pi/np.float(theta_number)
    R = Rp/np.float(factor)
    dim = int((R+1.15*h)/param)
    print h, R, dim, param
    from pytransfert import atmospectre
    wl_sample_micron = 1./bande_sample*1.0e+4
    ind, = np.where(wl_sample_micron <= wl)
    i_bande = ind[0]
    print 'Selected wavelength : %.3f'%(wl_sample_micron[i_bande])

    I_png = np.ones((2*dim,2*dim))
    I = I_tot[i_bande,:,:]

    bar = ProgressBar(2*dim+1,'Image generation')

    for i in range(-dim,dim) :

        for j in range(-dim,dim) :
            rho = np.sqrt((i*param)**2 + (j*param)**2)

            if rho <= R + h :

                if rho >= R :

                    theta = -math.atan2(i*param,j*param)

                    if theta >= 0 :
                        theta_line = int(round(theta/theta_step))
                    else :
                        theta_line = theta_number + int(round(theta/theta_step))
                        if theta_line == theta_number :
                            theta_line = 0

                    r = rho - R
                    r_line = int(round(r/r_step))
                    I_png[i+dim,j+dim] = I[r_line,theta_line]

                else :

                    I_png[i+dim,j+dim] = 'nan'

        bar.animate(i+dim+1)

    x = np.arange(-dim*param,dim*param,param)
    y = -np.arange(-dim*param,dim*param,param)
    X,Y = np.meshgrid(x,y)
    Z = I_png
    R_eff_bar,R_e,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I_tot,bande_sample,R_s,Rp,r_step*factor,extra,trans,Kcorr,Middle)
    R_eff = (R_e[i_bande]/factor - Rp/factor + R)
    print(R_e[i_bande]/factor,Rp/factor,R_eff)

    plt.imshow(I_png, extent = [-dim*param*factor/1000.,dim*param*factor/1000.,-dim*param*factor/1000.,dim*param*factor/1000.])
    plt.colorbar()
    lev = np.array([0,np.exp(-1),0.73,0.99])
    CS = plt.contour(X*factor/1000.,Y*factor/1000.,Z,levels=lev,colors='k')
    plt.clabel(CS,frontsize = 3, inline = 0)

    wh_R, = np.where((x >= -R-h)*(x <= R+h))
    line_b = np.sqrt(((R+h)**2 - (x[wh_R])**2))*factor/1000.
    line_black = np.zeros(line_b.size+2)
    line_black[1:line_black.size-1] = line_b

    wh_Reff, = np.where((x >= -R_eff)*(x <= R_eff))
    line_r = np.sqrt(((R_eff)**2 - (x[wh_Reff])**2))*factor/1000.
    line_red = np.zeros(line_r.size+2)
    line_red[1:line_red.size-1] = line_r

    x_Reff = np.ones(line_red.size)
    x_Reff[0],x_Reff[x_Reff.size-1] = -R_eff,R_eff
    x_Reff[1:x_Reff.size-1] = x[wh_Reff]
    x_R = np.ones(line_black.size)
    x_R[0],x_R[x_R.size-1] = -R-h,R+h
    x_R[1:x_R.size-1] = x[wh_R]

    plt.plot(x_Reff*factor/1000.,line_red,'--w',linewidth = 3)
    plt.plot(x_Reff*factor/1000.,-line_red,'--w', linewidth = 3)
    plt.plot(x_R*factor/1000.,line_black,'--k',linewidth = 3)
    plt.plot(x_R*factor/1000.,-line_black,'--k', linewidth = 3)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")

    plt.show()

    return I_png


########################################################################################################################


def Mean1Dmaker(I_all,I,bande_sample,R_s,Rp,r_step,theta_number,extra,name,Kcorr,Load) :

    from pytransfert import atmospectre

    bar = ProgressBar(theta_number,'Mean computation')

    if Kcorr == True :
        dim_bande = bande_sample.size-1

        if Load == False :

            R_eff = np.zeros((dim_bande,theta_number))
            R_eff_bar = np.zeros((2*dim_bande,theta_number))

            for theta in range(theta_number) :

                I_theta = I_all[theta,:,:,:]
                R_eff_bar[:,theta],R_eff[:,theta],ratio_bar,ratR_bar,bande_bar = \
                    atmospectre(I_theta,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=True)
                bar.animate(theta+1)

            np.save("%s"%(name),R_eff)

        else :

            R_eff = np.load("%s"%(name))
            R_eff_bar = np.zeros((2*dim_bande,theta_number))

            for i in range(dim_bande) :
                R_eff_bar[2*i] = R_eff[i]
                R_eff_bar[2*i+1] = R_eff[i]

        R_mean = np.zeros(dim_bande)
        R_mean_bar = np.zeros(2*dim_bande)

        for i in range(dim_bande) :

            R_mean[i] = np.mean(R_eff[i,:])
            R_mean_bar[2*i] = np.mean(R_eff_bar[2*i,:])
            R_mean_bar[2*i+1] = np.mean(R_eff_bar[2*i+1,:])

        R_eff_bar2,R_eff2,ratio_bar,ratR_bar,bande_bar = atmospectre(I,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=True)

        plt.plot(np.log10(1/(100.*(bande_bar))),R_mean_bar/1000.,linewidth=1,label='Mean-1D spectrum')
        plt.plot(np.log10(1/(100.*(bande_bar))),R_eff_bar2/1000.,linewidth=1,label='3D spectrum')

    else :

        dim_bande = bande_sample.size

        if Load == False :
            R_eff = np.zeros((dim_bande,theta_number))

            for theta in range(theta_number) :

                I_theta = I_all[theta,:,:,:]
                R_eff_bar,R_eff[:,theta],ratio_bar,ratR_bar,bande_bar = \
                    atmospectre(I_theta,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=False)
                bar.animate(theta+1)

            np.save("%s"%(name),R_eff)

        else :

            R_eff = np.load("%s"%(name))

        R_mean = np.zeros(dim_bande)

        for i in range(dim_bande) :

            R_mean[i] = np.mean(R_eff[i,:])

        R_eff_bar2,R_eff2,ratio_bar,ratR_bar,bande_bar = atmospectre(I,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=False)

        plt.plot(np.log10(1/(100.*(bande_sample+0.0001))),R_mean/1000.,linewidth=1,label='Mean-1D spectrum')
        plt.plot(np.log10(1/(100.*(bande_sample+0.0001))),R_eff2/1000.,linewidth=1,label='3D spectrum')

        R_mean_bar = 0
        R_eff_bar = 0

    return R_eff,R_eff_bar,R_mean,R_mean_bar,bande_sample,bande_bar


########################################################################################################################
########################################################################################################################

"""
    PROFIL_CYLCART

    Cette fonction transpose une ligne de la grille cartesienne dans la grille cylindrique et produit un profil de
    transmittance. A partir de ce profil (qui est construit par estimation de la position des points de variations
    de la transmittance) elle genere un tableau aux dimensions necessaires pour remplir la zone de transmittance
    associee a l'atmosphere dans la grille cartesienne. Elle realise ces profils de chaque cote de la planete. L'
    interpolation utilisee est ici lineaire mais peut etre modifiee.

    La fonction retourne les deux tableaux a coller dans la grille cartesienne pour chaque cote de l'hemisphere. Cette
    technique permet d'interpoler la grille cylindrique sans en augmenter la resolution.

"""

########################################################################################################################
########################################################################################################################


def profil_cylcart(I,Rp,h,B,r_step,theta_step,theta_number,param,pas,Climu,Climd) :

    X_u = np.arange(Climd,Climu + param, param)
    X_d = np.arange(-Climu,-Climd+param, param)

    x_slice = np.arange(0,h,pas)

    if abs(B) < Rp :
        lim_ud = np.sqrt(Rp**2 - B**2)
    else :
        lim_ud = 0

    I_ref_u = np.array([])
    x_ref_u = np.array([])
    pond_u = np.array([])
    I_ref_d = np.array([])
    x_ref_d = np.array([])
    pond_d = np.array([])
    i = 0

    for x in x_slice :

        C_u = lim_ud + x
        theta_u = math.atan2(B,C_u)

        if theta_u >= 0 :

            i_theta_u = int(round(theta_u/theta_step))
            theta_d = np.pi - theta_u
            i_theta_d = int(round(theta_d/theta_step))

            if i_theta_u == theta_number :
                i_theta_u = 0

            if i_theta_d == theta_number :
                i_theta_d = 0

        if theta_u < 0 :

            i_theta_u = theta_number + int(round(theta_u/theta_step))
            theta_d = - np.pi - theta_u
            i_theta_d = theta_number + int(round(theta_d/theta_step))

            if i_theta_u == theta_number :
                i_theta_u = 0

            if i_theta_d == theta_number :
                i_theta_d = 0

        if x == 0 :
            i_z = 0
        else :
            i_z = int(round((np.sqrt((lim_ud+x)**2 + B**2) - Rp)/float(r_step)))

        if x == x_slice[0] :

            i_theta_comp_u = i_theta_u
            i_theta_comp_d = i_theta_d

            i_z_comp_u = i_z
            i_z_comp_d = i_z

            deb_u = 0
            deb_d = 0

        else :

            if i_theta_u != i_theta_comp_u or i_z != i_z_comp_u :

                fin_u = i

                I_ref_u = np.append(I_ref_u, np.array(I[i_theta_comp_u,i_z_comp_d]))
                x_ref_u = np.append(x_ref_u, np.array(lim_ud + x_slice[int((fin_u+deb_u)/2.)]))
                pond_u = np.append(pond_u, np.array([fin_u - deb_u + 1]))

                deb_u = i + 1

                i_theta_comp_u = i_theta_u
                i_z_comp_u = i_z

            if i_theta_d != i_theta_comp_d or i_z != i_z_comp_d :

                fin_d = i

                I_ref_d = np.append(I_ref_d, np.array(I[i_theta_comp_d,i_z_comp_d]))
                x_ref_d = np.append(x_ref_d, np.array(-lim_ud - x_slice[int((fin_d+deb_d)/2.)]))
                pond_d = np.append(pond_d, np.array([fin_d - deb_d + 1]))

                deb_d = i + 1

                i_theta_comp_d = i_theta_d
                i_z_comp_d = i_z


        i = i + 1

    I_table_u = np.interp(X_u,x_ref_u,I_ref_u)
    I_table_d = np.interp(-X_d,-x_ref_d,I_ref_d)

    return I_table_d,I_table_u


########################################################################################################################


def repartition(axes,number_rank,rank,linear=False) :

    if linear == True :

        width_process = axes/number_rank
        diff = axes%number_rank
        if rank != number_rank-1 :
            if diff == 0 :
                dom_rank = np.linspace(rank*width_process,(rank+1)*width_process-1,width_process,dtype=np.int)
            else :
                if rank < diff :
                    width_process_fin = width_process + 1
                    deb = rank*width_process_fin
                    fin = (rank+1)*width_process_fin-1
                    dom_rank = np.linspace(deb,fin,width_process_fin,dtype=np.int)
                else :
                    width_process_fin = width_process + 1
                    deb = diff*width_process_fin+(rank-diff)*width_process
                    fin = diff*width_process_fin+(rank-diff+1)*width_process-1
                    dom_rank = np.linspace(deb,fin,width_process,dtype=np.int)
        else :
            if diff == 0 :
                dom_rank = np.linspace(rank*width_process,axes-1,axes-rank*width_process,dtype=np.int)
            else :
                width_process_fin = width_process + 1
                deb = diff*width_process_fin + (rank - diff)*width_process
                fin = axes - 1
                dom_rank = np.linspace(deb,fin,axes - deb,dtype=np.int)

    else :

        diff = axes%number_rank
        width_process = axes/number_rank
        dom_rank = np.arange(rank,axes,number_rank,dtype=np.int)

    return dom_rank


########################################################################################################################
########################################################################################################################


"""
    PYGCMTREAT_TOOLS

    Cette fonction transpose une ligne de la grille cartesienne dans la grille cylindrique et produit un profil de
    transmittance. A partir de ce profil (qui est construit par estimation de la position des points de variations
    de la transmittance) elle genere un tableau aux dimensions necessaires pour remplir la zone de transmittance
    associee a l'atmosphere dans la grille cartesienne. Elle realise ces profils de chaque cote de la planete. L'
    interpolation utilisee est ici lineaire mais peut etre modifiee.

    La fonction retourne les deux tableaux a coller dans la grille cartesienne pour chaque cote de l'hemisphere. Cette
    technique permet d'interpole la grille cylindrique sans en augmenter la resolution.

"""

########################################################################################################################
########################################################################################################################


def order_assign(z,p,q,k,ind,assist):

    order = np.zeros(6,dtype=np.int)
    ind_assist = np.zeros(3)
    dec = 0
    if assist[0] == 1 :
        ind_assist[0] = ind[0]
        dec = 1
    if assist[1] == 1 :
        ind_assist[1] = ind[0+dec]
        dec += 1
    if assist[2] == 1 :
        ind_assist[2] = ind[0+dec]

    order[0] = z
    order[1] = p
    order[2] = q
    order[3] = k-1 + ind_assist[0]
    order[4] = k-1 + ind_assist[1]
    order[5] = k-1 + ind_assist[2]

    return order


########################################################################################################################


def latlongalt(Rp,h,r,rho,r_step,z_level,delta,delta_step,reso_lat,alpha,alpha_o_ref,alpha_o_ref_0,alpha_step,reso_long,phi_obli,\
               x,x_range,x_reso,x_step,theta_range,theta_number,begin,inv,refrac,\
               Theta_init=False,Middle=True,Obliquity=False) :

    if Theta_init == True :

        if Middle == False :
            if (rho - Rp) >= z_level[1] :
                # Si r est compris entre z_step et le toit de l'atmosphere
                if (rho - Rp) < h - r_step/2. :
                    wh, = np.where(z_level > (rho - Rp))
                    if (rho - Rp) - z_level[wh[0]-1] <= r_step/2. :
                        z = wh[0]-1
                    else :
                        z = wh[0]
                else :
                    # Si le point est entre le toit et le milieu de la derniere couche
                    z = z_level.size - 1
            else :
                # Si le point est compris dans la premiere couche
                if (rho - Rp) <= z_level[1]/2. :
                    z = 0
                else :
                    z = 1
        else :
            # z_level donne l'echelle d'altitude et les niveaux inter-couches, sur une maille Middle
            # l'indice 0 correspond a la surface et les autres aux milieux de couche, en cherchant
            # l'indice pour lesquel z_level est superieur a r, on identifie directement l'indice
            # correspondant dans la maille Middle
            if (rho - Rp) < h :
                wh, = np.where(z_level > (rho - Rp))
                z = wh[0]
            else :
                z = z_level.size - 1
    else :
        z = 'nan'

    if Obliquity == False :

        # A partir de la latitude on en deduit l'indice p correpondant dans la maille spherique, cet indice
        # doit evoluer entre 0 (-pi/2) et reso_lat (pi/2)

        if delta%delta_step < delta_step/2. :
            delta_norm = delta - delta%delta_step
            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))
        else :
            delta_norm = delta - delta%delta_step + delta_step
            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))

        # A partir de la longitude, on en deduit l'indice q correspondant dans la maille spherique, cet
        # indice doit evoluer entre 0 (alpha -pi) et reso_long (alpha pi), sachant que le premier et le
        # dernier point sont identiques

        if alpha%alpha_step < alpha_step/2. :
            alpha_norm = alpha - alpha%alpha_step
            q = reso_long/2 + int(round(alpha_norm*reso_long/(2*np.pi)))
        else :
            alpha_norm = alpha - alpha%alpha_step + alpha_step
            q = reso_long/2 + int(round(alpha_norm*reso_long/(2*np.pi)))

        if q == reso_long :
            q = 0

    if Obliquity == True :

        delta_o = np.arcsin(np.sin(delta)*np.cos(phi_obli)+np.cos(delta)*np.sin(phi_obli)*np.sin(alpha))

        if np.abs(phi_obli) != np.pi/2. or begin == x_range :
            acos = np.cos(delta)*np.cos(alpha)/np.cos(delta_o)
            if acos < -1.0 or acos > 1.0 :
                print 'Check the longitude for theta %i : value that abort the process : '%(theta_range), \
                np.cos(delta)*np.cos(alpha)/np.cos(delta_o)
                if acos < -1.0 :
                    acos = -1.0
                else :
                    acos = 1.0
            alpha_o = np.arccos(acos)
        else :
            alpha_o = alpha_o_ref

        if np.abs(phi_obli) < np.pi/2. :
            x_ref = r*np.tan(np.abs(phi_obli))
        else :
            if np.abs(phi_obli) == np.pi :
                x_ref = 0.
            else :
                x_ref = r*np.tan(np.pi-np.abs(phi_obli))

        q, alpha_o_ref, alpha_o_ref_0, inv, refrac, begin = qoblicator(theta_range,x,x_range,x_ref,theta_number,phi_obli,\
                                        alpha_o,alpha_o_ref,alpha_o_ref_0,alpha_step,reso_long,inv,refrac,begin)

        if delta_o%delta_step < delta_step/2. :
            delta_norm = delta_o - delta_o%delta_step
            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))
        else :
            delta_norm = delta_o - delta_o%delta_step + delta_step
            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))

    return p, q, z, alpha_o_ref, alpha_o_ref_0, inv, refrac, begin


########################################################################################################################


def qoblicator(theta_range,x,x_range,x_ref,theta_number,phi_obli,alpha_o,alpha_o_ref,alpha_o_ref_0,alpha_step,reso_long,inv,refrac,begin) :

    if theta_range < theta_number/4 or theta_range > 3*theta_number/4 :
        # la longitude equatoriale est comprise entre pi/2 et 3pi/2 tandis que l'angle de reference des
        # donnees GCM est compris entre 0 et pi
        # l'angle calcule ne peut depasser pi, donc il faut lui specifier dans quelle tranche il se trouve

        if np.abs(phi_obli) < np.pi/2. :

            if alpha_o > alpha_o_ref :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
            elif alpha_o == alpha_o_ref :
                if inv == 0 :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
                else :
                    alpha_o_ref = alpha_o
            else :
                alpha_o_ref = alpha_o
                if x_range == begin+1 : refrac = 1

        if phi_obli == np.pi/2. :
            if theta_range <= theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if phi_obli == - np.pi/2. :
            if theta_range > 3*theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if np.abs(phi_obli) > np.pi/2. :

            if alpha_o > alpha_o_ref : alpha_o_ref = alpha_o
            elif alpha_o == alpha_o_ref :
                if inv == 0 : alpha_o_ref = alpha_o
                else :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
            else :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
                if x_range == begin+1 : refrac = 1

    if theta_range == theta_number/4. :

        if phi_obli > 0 :
            if phi_obli < np.pi/2. :
                if x < x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli == np.pi/2. : alpha_o = 2*np.pi - alpha_o
                else :
                    if x > -x_ref : alpha_o = 2*np.pi - alpha_o
        else :
            if phi_obli < -np.pi/2. :
                if x > -x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli != -np.pi/2. :
                    if x < x_ref : alpha_o = 2*np.pi - alpha_o

    if theta_range > theta_number/4. and theta_range < 3*theta_number/4. :
        # la longitude equatoriale est comprise entre 3pi/2 et pi/2 tandis que l'angle de reference des
        # donnees GCM est compris entre -pi et 0

        # l'angle calcule ne peut depasser pi, donc il faut lui specifier dans quelle tranche il se trouve

        if np.abs(phi_obli) < np.pi/2. :

            if alpha_o < alpha_o_ref :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
                if x_range == begin+1 : refrac = 1
            elif alpha_o == alpha_o_ref :
                if inv == 0 :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
                else : alpha_o_ref = alpha_o
            else : alpha_o_ref = alpha_o

        if phi_obli == np.pi/2. :
            if theta_range > theta_number/4. and theta_range < theta_number/2. : alpha_o = 2*np.pi - alpha_o

        if phi_obli == - np.pi/2. :
            if theta_range > theta_number/2. and theta_range <= 3*theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if np.abs(phi_obli) > np.pi/2. :

            if alpha_o < alpha_o_ref :
                alpha_o_ref = alpha_o
                if x_range == begin+1 : refrac = 1
            elif alpha_o == alpha_o_ref :
                if inv == 0 : alpha_o_ref = alpha_o
                else :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
            else :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o

    if theta_range == 3*theta_number/4. :

        if phi_obli > 0 :
            if phi_obli < np.pi/2. :
                if x < -x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli != np.pi/2. :
                    if x > x_ref : alpha_o = 2*np.pi - alpha_o
        else :
            if phi_obli < -np.pi/2. :
                if x > x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli == np.pi/2. : alpha_o = 2*np.pi - alpha_o
                else :
                    if x < -x_ref : alpha_o = 2*np.pi - alpha_o

    if x_range == begin : alpha_o_ref_0 = alpha_o

    if x_range != begin and x_range != begin+1 :

        if alpha_o%alpha_step < alpha_step/2. :
            alpha_norm = alpha_o - alpha_o%alpha_step
        else :
            alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

        if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
            q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
        if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
            q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

        if q_o == reso_long :
            q_o = 0

    else :

        if phi_obli == np.pi/2. :

            if alpha_o%alpha_step < alpha_step/2. :
                alpha_norm = alpha_o - alpha_o%alpha_step
            else :
                alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

            if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
            if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

            if q_o == reso_long :
                q_o = 0

        else :

            if x_range == begin + 1 :

                if refrac == 1 : alpha_o_ref_0 = 2*np.pi - alpha_o_ref_0

                if alpha_o%alpha_step < alpha_step/2. :
                    alpha_norm = alpha_o - alpha_o%alpha_step
                else :
                    alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

                if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                    q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
                if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                    q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

                if q_o == reso_long : q_o = 0

                q = np.array([q_o])

                if alpha_o_ref_0%alpha_step < alpha_step/2. :
                    alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step
                else :
                    alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step + alpha_step

                if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                    q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
                if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                    q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

                if q_o == reso_long : q_o = 0

                q = np.append(q,np.array([q_o]))
                q_o = q
                begin = -1

            else :

                q_o = 0

    return q_o, alpha_o_ref, alpha_o_ref_0, inv, refrac, begin


########################################################################################################################


def qoblicator_neg(theta_range,x,x_range,x_ref,theta_number,phi_obli,alpha_o,alpha_o_ref,alpha_step,reso_long,inv,refrac,begin) :

    if theta_range < theta_number/4 or theta_range > 3*theta_number/4 :
        # la longitude equatoriale est comprise entre pi/2 et 3pi/2 tandis que l'angle de reference des
        # donnees GCM est compris entre 0 et pi
        # l'angle calcule ne peut depasser pi, donc il faut lui specifier dans quelle tranche il se trouve

        if np.abs(phi_obli) < np.pi/2. :

            if alpha_o > alpha_o_ref :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
            elif alpha_o == alpha_o_ref :
                if inv == 0 :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
                else :
                    alpha_o_ref = alpha_o
            else :
                alpha_o_ref = alpha_o
                if x_range == begin+1 : refrac = 1

        if phi_obli == np.pi/2. :
            if theta_range <= theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if phi_obli == - np.pi/2. :
            if theta_range > 3*theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if np.abs(phi_obli) > np.pi/2. :

            if alpha_o > alpha_o_ref : alpha_o_ref = alpha_o
            elif alpha_o == alpha_o_ref :
                if inv == 0 : alpha_o_ref = alpha_o
                else :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
            else :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
                if x_range == begin+1 : refrac = 1


    if theta_range == theta_number/4. :

        if phi_obli > 0 :
            if phi_obli < np.pi/2. :
                if x < x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli == np.pi/2. : alpha_o = 2*np.pi - alpha_o
                else :
                    if x > -x_ref : alpha_o = 2*np.pi - alpha_o
        else :
            if phi_obli < -np.pi/2. :
                if x < -x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli != -np.pi/2. :
                    if x > x_ref : alpha_o = 2*np.pi - alpha_o


    if theta_range > theta_number/4. and theta_range < 3*theta_number/4. :
        # la longitude equatoriale est comprise entre 3pi/2 et pi/2 tandis que l'angle de reference des
        # donnees GCM est compris entre -pi et 0
        # l'angle calcule ne peut depasser pi, donc il faut lui specifier dans quelle tranche il se trouve

        if np.abs(phi_obli) < np.pi/2. :

            if alpha_o < alpha_o_ref :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o
                if x_range == begin+1 : refrac = 1
            elif alpha_o == alpha_o_ref :
                if inv == 0 :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
                else : alpha_o_ref = alpha_o
            else : alpha_o_ref = alpha_o

        if phi_obli == np.pi/2. :
            if theta_range > theta_number/4. and theta_range < theta_number/2. : alpha_o = 2*np.pi - alpha_o

        if phi_obli == - np.pi/2. :
            if theta_range > theta_number/2. and theta_range <= 3*theta_number/4. : alpha_o = 2*np.pi - alpha_o

        if np.abs(phi_obli) > np.pi/2. :

            if alpha_o < alpha_o_ref :
                alpha_o_ref = alpha_o
                if x_range == begin+1 : refrac = 1
            elif alpha_o == alpha_o_ref :
                if inv == 0 : alpha_o_ref = alpha_o
                else :
                    alpha_o_ref = alpha_o
                    alpha_o = 2*np.pi - alpha_o
            else :
                alpha_o_ref = alpha_o
                alpha_o = 2*np.pi - alpha_o

    if theta_range == 3*theta_number/4. :

        if phi_obli > 0 :
            if phi_obli < np.pi/2. :
                if x < -x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli != np.pi/2. :
                    if x > x_ref : alpha_o = 2*np.pi - alpha_o
        else :
            if phi_obli < -np.pi/2. :
                if x < x_ref : alpha_o = 2*np.pi - alpha_o
            else :
                if phi_obli == np.pi/2. : alpha_o = 2*np.pi - alpha_o
                else :
                    if x > -x_ref : alpha_o = 2*np.pi - alpha_o

    if x_range == begin : alpha_o_ref_0 = alpha_o

    if x_range != begin and x_range != begin+1 :

        if alpha_o%alpha_step < alpha_step/2. :
            alpha_norm = alpha_o - alpha_o%alpha_step
        else :
            alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

        if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
            q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
        if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
            q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

        if q_o == reso_long :
            q_o = 0

    else :

        if phi_obli == np.pi/2. :

            if alpha_o%alpha_step < alpha_step/2. :
                alpha_norm = alpha_o - alpha_o%alpha_step
            else :
                alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

            if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
            if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

            if q_o == reso_long :
                q_o = 0

        else :

            if x_range == begin+1 :

                if refrac == 1 : alpha_o_ref_0 = 2*np.pi - alpha_o_ref_0

                if alpha_o%alpha_step < alpha_step/2. :
                    alpha_norm = alpha_o - alpha_o%alpha_step
                else :
                    alpha_norm = alpha_o - alpha_o%alpha_step + alpha_step

                if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                    q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
                if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                    q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

                if q_o == reso_long : q_o = 0

                q = np.array([q_o])

                if alpha_o_ref_0%alpha_step < alpha_step/2. :
                    alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step
                else :
                    alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step + alpha_step

                if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                    q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
                if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                    q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

                if q_o == reso_long : q_o = 0

                q = np.append(q,np.array([q_o]))
                q_o = q
                begin = -1

    return q_o, alpha_o_ref, inv, refrac, begin


########################################################################################################################
########################################################################################################################


def vap_sat(T) :

    T_0, r_2, r_3, r_4, P = 273.16, 661.14, np.array([17.269,21.875]), np.array([35.86,7.66]), 0.

    if T > 647. :
        P = 1.e+10
    if T < 100. :
        P = 0.
    if T > T_0 and T <= 647. :
        P = r_2*np.exp(r_3[0]*((T-T_0)/(T-r_4[0])))
    if T > 100. and T <= T_0 :
        P = r_2*np.exp(r_3[1]*((T-T_0)/(T-r_4[1])))

    return P


########################################################################################################################


def interpolation(x,x_sample,grid) :

    wh_x, = np.where(x_sample > x)
    if wh_x.size == 0 :
        x_u,x_d = x_sample.size-1,x_sample.size-1
        c1,c2 = 0.,1.
    else :
        if wh_x[0] == 0 :
            x_u,x_d = 0,0
            c1,c2 = 1.,0.
        else :
            x_u,x_d = wh_x[0],wh_x[0]-1
            c1 = (x-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
            c2 = (x_sample[x_u]-x)/(x_sample[x_u]-x_sample[x_d])

    res = c1*grid[x_u]+c2*grid[x_d]
    c_grid = np.array([c1,c2])
    i_grid = np.array([x_d,x_u])

    return res, c_grid, i_grid


########################################################################################################################


def interpolation_multi(x,x_sample,grid) :

    x_size = x.size
    c_grid = np.zeros((x_size,2),dtype=np.float64)
    i_grid = np.zeros((x_size,2),dtype=np.int)
    res = np.zeros(x_size,dtype=np.float64)

    for i_x in range(x_size) :
        X = x[i_x]
        wh_x, = np.where(x_sample > X)
        if wh_x.size == 0 :
            x_u,x_d = x_sample.size-1,x_sample.size-1
            c1,c2 = 0.,1.
        else :
            if wh_x[0] == 0 :
                x_u,x_d = 0,0
                c1,c2 = 1.,0.
            else :
                x_u,x_d = wh_x[0],wh_x[0]-1
                c1 = (X-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
                c2 = (x_sample[x_u]-X)/(x_sample[x_u]-x_sample[x_d])
        c_grid[i_x,:] = np.array([c1,c2])
        i_grid[i_x,:] = np.array([x_d,x_u])

        res[i_x] = c1*grid[x_u]+c2*grid[x_d]

    return res, c_grid, i_grid


########################################################################################################################


def interp2olation(x,y,x_sample,y_sample,grid) :

    x_size = x.size
    res = np.zeros(x_size,dtype=np.float64)
    c_grid = np.zeros((x_size,4),dtype=np.float64)
    i_grid = np.zeros((x_size,4),dtype=np.float64)

    for i_x in range(x_size) :

        X = x[i_x]
        Y = y[i_x]

        wh_x, = np.where(x_sample > X)

        if wh_x.size == 0 :
            x_u,x_d = x_sample.size-1,x_sample.size-1
            c1,c2 = 0.,1.
        else :
            if wh_x[0] == 0 :
                x_u,x_d = 0,0
                c1,c2 = 1.,0.
            else :
                x_u,x_d = wh_x[0],wh_x[0]-1
                c1 = (X-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
                c2 = (x_sample[x_u]-X)/(x_sample[x_u]-x_sample[x_d])

        wh_y, = np.where(y_sample > Y)

        if wh_y.size == 0 :
            y_u,y_d = y_sample.size-1,y_sample.size-1
            c3,c4 = 0.,1.
        else :
            if wh_y[0] == 0 :
                y_u,y_d = 0,0
                c3,c4 = 1.,0.
            else :
                y_u,y_d = wh_y[0],wh_y[0]-1
                c3 = (Y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
                c4 = (y_sample[y_u]-Y)/(y_sample[y_u]-y_sample[y_d])

        res_1 = c1*grid[x_u,y_u]+c2*grid[x_d,y_u]
        res_2 = c1*grid[x_u,y_d]+c2*grid[x_d,y_d]
        res[i_x] = c3*res_1+c4*res_2

        c_grid[i_x,:] = np.array([c1,c2,c3,c4])
        i_grid[i_x,:] = np.array([x_d,x_u,y_d,y_u])

    return res,c_grid,i_grid


########################################################################################################################


def interp2olation_uni(x,y,x_sample,y_sample,grid) :

    wh_x, = np.where(x_sample > x)

    if wh_x.size == 0 :
        x_u,x_d = x_sample.size-1,x_sample.size-1
        c1,c2 = 0.,1.
    else :
        if wh_x[0] == 0 :
            x_u,x_d = 0,0
            c1,c2 = 1.,0.
        else :
            x_u,x_d = wh_x[0],wh_x[0]-1
            c1 = (x-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
            c2 = (x_sample[x_u]-x)/(x_sample[x_u]-x_sample[x_d])

    wh_y, = np.where(y_sample > y)

    if wh_y.size == 0 :
        y_u,y_d = y_sample.size-1,y_sample.size-1
        c3,c4 = 0.,1.
    else :
        if wh_y[0] == 0 :
            y_u,y_d = 0,0
            c3,c4 = 1.,0.
        else :
            y_u,y_d = wh_y[0],wh_y[0]-1
            c3 = (y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
            c4 = (y_sample[y_u]-y)/(y_sample[y_u]-y_sample[y_d])

    res_1 = c1*grid[x_u,y_u]+c2*grid[x_d,y_u]
    res_2 = c1*grid[x_u,y_d]+c2*grid[x_d,y_d]
    res = c3*res_1+c4*res_2

    c_grid= np.array([c1,c2,c3,c4])
    i_grid = np.array([x_d,x_u,y_d,y_u])

    return res,c_grid,i_grid


########################################################################################################################


def interp2olation_uni_multi(x,y,x_sample,y_sample,grid) :

    sh_gr = np.shape(grid)
    dim = sh_gr[0]
    res_1 = np.zeros(dim,dtype=np.float64)
    res_2 = np.zeros(dim,dtype=np.float64)
    res = np.zeros(dim,dtype=np.float64)

    wh_x, = np.where(x_sample > x)

    if wh_x.size == 0 :
        x_u,x_d = x_sample.size-1,x_sample.size-1
        c1,c2 = 0.,1.
    else :
        if wh_x[0] == 0 :
            x_u,x_d = 0,0
            c1,c2 = 1.,0.
        else :
            x_u,x_d = wh_x[0],wh_x[0]-1
            c1 = (x-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
            c2 = (x_sample[x_u]-x)/(x_sample[x_u]-x_sample[x_d])

    wh_y, = np.where(y_sample > y)

    if wh_y.size == 0 :
        y_u,y_d = y_sample.size-1,y_sample.size-1
        c3,c4 = 0.,1.
    else :
        if wh_y[0] == 0 :
            y_u,y_d = 0,0
            c3,c4 = 1.,0.
        else :
            y_u,y_d = wh_y[0],wh_y[0]-1
            c3 = (y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
            c4 = (y_sample[y_u]-y)/(y_sample[y_u]-y_sample[y_d])

    c_grid= np.array([c1,c2,c3,c4])
    i_grid = np.array([x_d,x_u,y_d,y_u])

    for i_dim in range(dim) :
        res_1[i_dim] = c1*grid[i_dim,x_u,y_u]+c2*grid[i_dim,x_d,y_u]
        res_2[i_dim] = c1*grid[i_dim,x_u,y_d]+c2*grid[i_dim,x_d,y_d]
        res[i_dim] = c3*res_1[i_dim]+c4*res_2[i_dim]

    return res,c_grid,i_grid


########################################################################################################################


def interp2olation_multi(x,y,x_sample,y_sample,grid) :

    x_size = x.size
    sh_gr = np.shape(grid)
    dim = sh_gr[0]
    c_grid = np.zeros((x_size,4),dtype=np.float64)
    i_grid = np.zeros((x_size,4),dtype=np.int)
    res_int = np.zeros((2,dim,x_size),dtype=np.float64)
    res = np.zeros((dim,x_size),dtype=np.float64)

    for i_x in range(x_size) :

        X = x[i_x]
        Y = y[i_x]

        wh_x, = np.where(x_sample > X)

        if wh_x.size == 0 :
            x_u,x_d = x_sample.size-1,x_sample.size-1
            c1,c2 = 0.,1.
        else :
            if wh_x[0] == 0 :
                x_u,x_d = 0,0
                c1,c2 = 1.,0.
            else :
                x_u,x_d = wh_x[0],wh_x[0]-1
                c1 = (X-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
                #c2 = (x_sample[x_u]-X)/(x_sample[x_u]-x_sample[x_d])
                c2 = 1.-c1

        wh_y, = np.where(y_sample > Y)

        if wh_y.size == 0 :
            y_u,y_d = y_sample.size-1,y_sample.size-1
            c3,c4 = 0.,1.
        else :
            if wh_y[0] == 0 :
                y_u,y_d = 0,0
                c3,c4 = 1.,0.
            else :
                y_u,y_d = wh_y[0],wh_y[0]-1
                c3 = (Y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
                #c4 = (y_sample[y_u]-Y)/(y_sample[y_u]-y_sample[y_d])
                c4 = 1.-c3

        c_grid[i_x,:] = np.array([c1,c2,c3,c4])
        i_grid[i_x,:] = np.array([x_d,x_u,y_d,y_u])

    for i_dim in range(dim) :
        res_int[0,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,3]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,3]]*c_grid[:,1]
        res_int[1,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,2]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,2]]*c_grid[:,1]

        res[i_dim] = res_int[0,i_dim]*c_grid[:,2]+res_int[1,i_dim]*c_grid[:,3]

    return res,c_grid,i_grid


########################################################################################################################


def interp3olation(x,y,z,x_sample,y_sample,z_sample,grid) :

    x_size = x.size
    c_grid = np.zeros((x_size,6),dtype=np.float64)
    i_grid = np.zeros((x_size,6),dtype=np.int)
    res_int = np.zeros((4,x_size),dtype=np.float64)

    for i_x in range(x_size) :

        X = x[i_x]
        Y = y[i_x]
        Z = z[i_x]

        wh_x, = np.where(x_sample > X)

        if wh_x.size == 0 :
            x_u,x_d = x_sample.size-1,x_sample.size-1
            c1,c2 = 0.,1.
        else :
            if wh_x[0] == 0 :
                x_u,x_d = 0,0
                c1,c2 = 1.,0.
            else :
                x_u,x_d = wh_x[0],wh_x[0]-1
                c1 = (X-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
                #c2 = (x_sample[x_u]-X)/(x_sample[x_u]-x_sample[x_d])
                c2 = 1.-c1

        wh_y, = np.where(y_sample > Y)

        if wh_y.size == 0 :
            y_u,y_d = y_sample.size-1,y_sample.size-1
            c3,c4 = 0.,1.
        else :
            if wh_y[0] == 0 :
                y_u,y_d = 0,0
                c3,c4 = 1.,0.
            else :
                y_u,y_d = wh_y[0],wh_y[0]-1
                c3 = (Y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
                #c4 = (y_sample[y_u]-Y)/(y_sample[y_u]-y_sample[y_d])
                c4 = 1.-c3

        wh_z, = np.where(z_sample > Z)

        if wh_z.size == 0 :
            z_u,z_d = x_sample.size-1,x_sample.size-1
            c5,c6 = 0.,1.
        else :
            if wh_z[0] == 0 :
                z_u,z_d = 0,0
                c5,c6 = 1.,0.
            else :
                z_u,z_d = wh_z[0],wh_z[0]-1
                c5 = (Z-z_sample[z_d])/(z_sample[z_u]-z_sample[z_d])
                #c6 = (z_sample[z_u]-Z)/(z_sample[z_u]-z_sample[z_d])
                c6 = 1.-c5

        c_grid[i_x,:] = np.array([c1,c2,c3,c4,c5,c6])
        i_grid[i_x,:] = np.array([x_d,x_u,y_d,y_u,z_d,z_u])

    res_int[0] = grid[i_grid[:,1],i_grid[:,3],i_grid[:,5]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,3],i_grid[5]]*c_grid[:,1]
    res_int[1] = grid[i_grid[:,1],i_grid[:,2],i_grid[:,5]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,2],i_grid[5]]*c_grid[:,1]
    res_int[2] = grid[i_grid[:,1],i_grid[:,3],i_grid[:,4]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,3],i_grid[4]]*c_grid[:,1]
    res_int[3] = grid[i_grid[:,1],i_grid[:,2],i_grid[:,4]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,2],i_grid[4]]*c_grid[:,1]
    res_int_zu = res_int[0]*c_grid[:,2]+res_int[1]*c_grid[:,3]
    res_int_zd = res_int[2]*c_grid[:,2]+res_int[3]*c_grid[:,3]
    res = res_int_zu*c_grid[:,4]+res_int_zd*c_grid[:,5]

    return res,c_grid,i_grid


########################################################################################################################


def interp3olation_uni(x,y,z,x_sample,y_sample,z_sample,grid) :

    res_int = np.zeros(4,dtype=np.float64)

    wh_x, = np.where(x_sample > x)

    if wh_x.size == 0 :
        x_u,x_d = x_sample.size-1,x_sample.size-1
        c1,c2 = 0.,1.
    else :
        if wh_x[0] == 0 :
            x_u,x_d = 0,0
            c1,c2 = 1.,0.
        else :
            x_u,x_d = wh_x[0],wh_x[0]-1
            c1 = (x-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
            #c2 = (x_sample[x_u]-x)/(x_sample[x_u]-x_sample[x_d])
            c2 = 1.-c1

    wh_y, = np.where(y_sample > y)

    if wh_y.size == 0 :
        y_u,y_d = y_sample.size-1,y_sample.size-1
        c3,c4 = 0.,1.
    else :
        if wh_y[0] == 0 :
            y_u,y_d = 0,0
            c3,c4 = 1.,0.
        else :
            y_u,y_d = wh_y[0],wh_y[0]-1
            c3 = (y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
            #c4 = (y_sample[y_u]-y)/(y_sample[y_u]-y_sample[y_d])
            c4 = 1.-c3

    wh_z, = np.where(z_sample > z)

    if wh_z.size == 0 :
        z_u,z_d = x_sample.size-1,x_sample.size-1
        c5,c6 = 0.,1.
    else :
        if wh_z[0] == 0 :
            z_u,z_d = 0,0
            c5,c6 = 1.,0.
        else :
            z_u,z_d = wh_z[0],wh_z[0]-1
            c5 = (z-z_sample[z_d])/(z_sample[z_u]-z_sample[z_d])
            #c6 = (z_sample[z_u]-Z)/(z_sample[z_u]-z_sample[z_d])
            c6 = 1.-c5

    c_grid = np.array([c1,c2,c3,c4,c5,c6])
    i_grid = np.array([x_d,x_u,y_d,y_u,z_d,z_u])

    res_int[0] = grid[i_grid[:,1],i_grid[:,3],i_grid[:,5]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,3],i_grid[5]]*c_grid[:,1]
    res_int[1] = grid[i_grid[:,1],i_grid[:,2],i_grid[:,5]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,2],i_grid[5]]*c_grid[:,1]
    res_int[2] = grid[i_grid[:,1],i_grid[:,3],i_grid[:,4]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,3],i_grid[4]]*c_grid[:,1]
    res_int[3] = grid[i_grid[:,1],i_grid[:,2],i_grid[:,4]]*c_grid[:,0]+grid[i_grid[:,0],i_grid[:,2],i_grid[4]]*c_grid[:,1]
    res_int_zu = res_int[0]*c_grid[:,2]+res_int[1]*c_grid[:,3]
    res_int_zd = res_int[2]*c_grid[:,2]+res_int[3]*c_grid[:,3]
    res = res_int_zu*c_grid[:,4]+res_int_zd*c_grid[:,5]

    return res,c_grid,i_grid


########################################################################################################################


def interp3olation_uni_multi(x,y,z,x_sample,y_sample,z_sample,grid) :

    sh_gr, = np.shape(grid)
    dim = sh_gr[0]
    res_int = np.zeros((dim,4),dtype=np.float64)

    wh_x, = np.where(x_sample > x)

    if wh_x.size == 0 :
        x_u,x_d = x_sample.size-1,x_sample.size-1
        c1,c2 = 0.,1.
    else :
        if wh_x[0] == 0 :
            x_u,x_d = 0,0
            c1,c2 = 1.,0.
        else :
            x_u,x_d = wh_x[0],wh_x[0]-1
            c1 = (x-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
            #c2 = (x_sample[x_u]-x)/(x_sample[x_u]-x_sample[x_d])
            c2 = 1.-c1

    wh_y, = np.where(y_sample > y)

    if wh_y.size == 0 :
        y_u,y_d = y_sample.size-1,y_sample.size-1
        c3,c4 = 0.,1.
    else :
        if wh_y[0] == 0 :
            y_u,y_d = 0,0
            c3,c4 = 1.,0.
        else :
            y_u,y_d = wh_y[0],wh_y[0]-1
            c3 = (y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
            #c4 = (y_sample[y_u]-y)/(y_sample[y_u]-y_sample[y_d])
            c4 = 1.-c3

    wh_z, = np.where(z_sample > z)

    if wh_z.size == 0 :
        z_u,z_d = x_sample.size-1,x_sample.size-1
        c5,c6 = 0.,1.
    else :
        if wh_z[0] == 0 :
            z_u,z_d = 0,0
            c5,c6 = 1.,0.
        else :
            z_u,z_d = wh_z[0],wh_z[0]-1
            c5 = (z-z_sample[z_d])/(z_sample[z_u]-z_sample[z_d])
            #c6 = (z_sample[z_u]-Z)/(z_sample[z_u]-z_sample[z_d])
            c6 = 1.-c5

    c_grid = np.array([c1,c2,c3,c4,c5,c6])
    i_grid = np.array([x_d,x_u,y_d,y_u,z_d,z_u])

    for i_dim in range(dim) :
        res_int[0,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,3],i_grid[:,5]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,3],i_grid[5]]*c_grid[:,1]
        res_int[1,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,2],i_grid[:,5]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,2],i_grid[5]]*c_grid[:,1]
        res_int[2,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,3],i_grid[:,4]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,3],i_grid[4]]*c_grid[:,1]
        res_int[3,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,2],i_grid[:,4]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,2],i_grid[4]]*c_grid[:,1]
        res_int[4,i_dim] = res_int[0]*c_grid[i_dim,:,2]+res_int[1]*c_grid[i_dim,:,3]
        res_int[5,i_dim] = res_int[2]*c_grid[i_dim,:,2]+res_int[3]*c_grid[i_dim,:,3]
        res = res_int[4,i_dim]*c_grid[i_dim,:,4]+res_int[5,i_dim]*c_grid[i_dim,:,5]

    return res,c_grid,i_grid


########################################################################################################################


def interp3olation_multi(x,y,z,x_sample,y_sample,z_sample,grid) :

    x_size = x.size
    sh_gr = np.shape(grid)
    dim = sh_gr[0]
    c_grid = np.zeros((x_size,6),dtype=np.float64)
    i_grid = np.zeros((x_size,6),dtype=np.int)
    res_int = np.zeros((4,dim,x_size),dtype=np.float64)
    res = np.zeros((dim,x_size),dtype=np.float64)

    for i_x in range(x_size) :

        X = x[i_x]
        Y = y[i_x]
        Z = z[i_x]

        wh_x, = np.where(x_sample > X)

        if wh_x.size == 0 :
            x_u,x_d = x_sample.size-1,x_sample.size-1
            c1,c2 = 0.,1.
        else :
            if wh_x[0] == 0 :
                x_u,x_d = 0,0
                c1,c2 = 1.,0.
            else :
                x_u,x_d = wh_x[0],wh_x[0]-1
                c1 = (X-x_sample[x_d])/(x_sample[x_u]-x_sample[x_d])
                #c2 = (x_sample[x_u]-X)/(x_sample[x_u]-x_sample[x_d])
                c2 = 1.-c1

        wh_y, = np.where(y_sample > Y)

        if wh_y.size == 0 :
            y_u,y_d = y_sample.size-1,y_sample.size-1
            c3,c4 = 0.,1.
        else :
            if wh_y[0] == 0 :
                y_u,y_d = 0,0
                c3,c4 = 1.,0.
            else :
                y_u,y_d = wh_y[0],wh_y[0]-1
                c3 = (Y-y_sample[y_d])/(y_sample[y_u]-y_sample[y_d])
                #c4 = (y_sample[y_u]-Y)/(y_sample[y_u]-y_sample[y_d])
                c4 = 1.-c3

        wh_z, = np.where(z_sample > Z)

        if wh_z.size == 0 :
            z_u,z_d = x_sample.size-1,x_sample.size-1
            c5,c6 = 0.,1.
        else :
            if wh_z[0] == 0 :
                z_u,z_d = 0,0
                c5,c6 = 1.,0.
            else :
                z_u,z_d = wh_z[0],wh_z[0]-1
                c5 = (Z-z_sample[z_d])/(z_sample[z_u]-z_sample[z_d])
                #c6 = (z_sample[z_u]-Z)/(z_sample[z_u]-z_sample[z_d])
                c6 = 1.-c5

        c_grid[i_x,:] = np.array([c1,c2,c3,c4,c5,c6])
        i_grid[i_x,:] = np.array([x_d,x_u,y_d,y_u,z_d,z_u])

    for i_dim in range(dim) :
        res_int[0,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,3],i_grid[:,5]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,3],i_grid[5]]*c_grid[:,1]
        res_int[1,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,2],i_grid[:,5]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,2],i_grid[5]]*c_grid[:,1]
        res_int[2,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,3],i_grid[:,4]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,3],i_grid[4]]*c_grid[:,1]
        res_int[3,i_dim] = grid[i_dim,i_grid[:,1],i_grid[:,2],i_grid[:,4]]*c_grid[:,0]+grid[i_dim,i_grid[:,0],i_grid[:,2],i_grid[4]]*c_grid[:,1]
        res_int_zu = res_int[0,i_dim]*c_grid[:,2]+res_int[1,i_dim]*c_grid[:,3]
        res_int_zd = res_int[2,i_dim]*c_grid[:,2]+res_int[3,i_dim]*c_grid[:,3]
        res[i_dim] = res_int_zu*c_grid[:,4]+res_int_zd*c_grid[:,5]

    return res,c_grid,i_grid


########################################################################################################################
########################################################################################################################


def chain_pressure(iter_1,iter_2,it_1,it_2,fac,redond,redond_1,redond_2,R_mean,Rp_o,p_surf) :

    if iter_1[0] == 0 :
        iter_1 = np.array([R_mean-Rp_o,p_surf])
        fac = 1
    else :
        if iter_2[0] == 0 :
            iter_2 = iter_1
            iter_1 = np.array([R_mean-Rp_o,p_surf])
            redond += 2.
        else :
            iter_1 = np.array([R_mean-Rp_o,p_surf])
            iter_2 = np.array([0])

    if redond < 13. :
        if iter_1[0] != 0 and iter_2[0] != 0 :
            delta = (iter_2[0] - iter_1[0])/(iter_2[1] - iter_1[1])
            cons = iter_2[0] - delta*iter_2[1]
            p_surf = -cons/delta
            print 'Input correction'
        else :
            if R_mean > Rp_o :
                p_surf = (1-0.05/redond)*p_surf
            else :
                p_surf = (1+0.05/redond)*p_surf
    else :
        if R_mean > Rp_o :
            if it_1 == 1 :
                fac -= 1
                it_2 = 3
            else :
                fac -= 1./np.float(it_1)
                if redond_1 == 0 :
                    it_2 = it_1**1.3
                    redond_1, redond_2 = 1, 0
        else :
            if it_2 == 1 :
                fac += 1
                it_1 = 2
            else :
                fac += 1./np.float(it_2)
                if redond_2 == 0 :
                    it_1 = it_2**1.3
                    redond_1, redond_2 = 0, 1

    P_surf = 10**(p_surf)

    return iter_1,iter_2,it_1,it_2,redond,redond_1,redond_2,p_surf,P_surf,fac


########################################################################################################################


try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = True

class ProgressBar:
    def __init__(self, iterations,name):
        self.iterations = iterations
        self.name = '%s' %name
        self.prog_bar = ''
        self.fill_char = '='
        self.width = 1
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython

    def animate_ipython(self, iter):
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        if elapsed_iter != self.iterations :
            self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)
        else :
            self.prog_bar += '  %d of %s complete\n' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))

        all_full = 20
        num_hashes = int((percent_done / 100.0) * all_full)
        pct_string = ' %d%%' % percent_done
        self.prog_bar = self.name + ' [' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']' + pct_string

    def __str__(self):
        return str(self.prog_bar)


########################################################################################################################


class ProgressBar2:
    def __init__(self, iterations, repetition, name):
        self.iterations = iterations
        self.repetition = repetition
        self.name = '%s' %name
        self.prog_bar = ''
        self.fill_char = '='
        self.width = 1
        self.__update_amount(0)
        if have_ipython:
            self.animate2 = self.animate2_ipython

    def animate2_ipython(self, iter, iter_rep):
        print '\r', self,
        sys.stdout.flush()
        self.update2_iteration(iter + 1, iter_rep + 1)

    def update2_iteration(self, elapsed_iter, elapsed_iter_rep):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        if elapsed_iter + self.repetition*self.iterations != self.iterations*self.repetition :
            self.prog_bar += '  %d of %s complete, repetition %d' % (elapsed_iter, self.iterations, elapsed_iter_rep)
        else :
            self.prog_bar += '  %d of %s complete, repetition %d\n' % (elapsed_iter, self.iterations, elapsed_iter_rep)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))

        all_full = 20
        num_hashes = int((percent_done / 100.0) * all_full)
        pct_string = ' %d%%' % percent_done
        self.prog_bar = self.name + ' [' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']' + pct_string

    def __str__(self):
        return str(self.prog_bar)


########################################################################################################################


def reverse(x) :
    x = np.array(x)
    k = x.size
    X = np.array([])
    for i in range(k) :
        X = np.append(X, [x[k - 1 - i]])
    return X


########################################################################################################################


def dataread(data) :
    lines = data.readlines()
    u = np.shape(lines)
    P = np.zeros(u[0])
    n = np.zeros(u[0])
    z = np.zeros(u[0])
    T = np.zeros(u[0])
    for i in range(u[0]) :
        P[u[0]-1-i] = float(lines[i][0:8])
        z[u[0]-1-i] = float(lines[i][9:17])
        n[u[0]-1-i] = float(lines[i][19:30])
        T[u[0]-1-i] = float(lines[i][31:37])
    return u,n,z,T,P


########################################################################################################################


def altitude(x,max,b,Rp) :
    zh = np.sqrt((Rp+b)**2 + x**2) - Rp
    return zh


########################################################################################################################


def section(s,T) :
    section = s*np.sqrt(293/float(T))
    return section


########################################################################################################################


def echelle(T,g,M) :
    H = R_gp*T/(M*g)
    return H


########################################################################################################################


def echellecor(T,g,M,Rp,z) :
    H = np.zeros(T.size)
    for i in range(T.size) :
        geff = g*Rp**2/((Rp+z[i])**2)
        H[i] = R_gp*T[i]/(M*geff)
    return H


########################################################################################################################


def nanarrsum(X,type) :

    a,b = np.shape(X)

    if type == "raw" :
        X_arr = np.zeros(a)
        for i in range(a) :
            X_arr[i] = np.nansum(X[i,:])

    if type == "col" :
        X_arr = np.zeros(b)
        for i in range(b) :
            X_arr[i] = np.nansum(X[:,i])

    return X_arr

