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

    Version : 6.0

    Date de derniere modification : 06.07.2016

    >> Insertion des fonctions de modulation de la densite et de la calibration

    Date de derniere modification : 12.12.2016

"""

########################################################################################################################
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

        '''
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))

        if percent_done%10==0:
            self.prog_bar += '='+self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

        else:
            self.prog_bar += self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])
        '''


    def __str__(self):
        return str(self.prog_bar)

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


def dataread (data) :
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


def altitude (x,max,b,Rp) :
    zh = np.sqrt((Rp+b)**2 + x**2) - Rp
    return zh


########################################################################################################################


def section (s,T) :
    section = s*np.sqrt(293/float(T))
    return section


########################################################################################################################


def echelle (T,g,M) :
    H = R_gp*T/(M*g)
    return H


########################################################################################################################


def echellecor (T,g,M,Rp,z) :
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

            if single == "KCl" :

                stu = np.append(stu,np.array(["cloud_KCl"]))

            if single == "ZnS" :

                stu = np.append(stu,np.array(["cloud_ZnS"]))

    else :

        if Continuum == True :

            stu = np.append(stu,np.array(["cont"]))

        if Scattering == True :

            stu = np.append(stu,np.array(["sca"]))

        if Clouds == True :

            if single == "no" :

                stu = np.append(stu,np.array(["cloud"]))

            if single == "KCl" :

                stu = np.append(stu,np.array(["cloud_KCl"]))

            if single == "ZnS" :

                stu = np.append(stu,np.array(["cloud_ZnS"]))

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
           domain,stud,lim_alt,rupt_alt,Discreet,Integration,Module,Optimal,Kcorr) :

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
        s_m = '%sI_%s_%.1f_%s_%i_%i%i_%i_%i%i_%s_%i_%.2f_%.2f_%s_%s.npy'\
        %(save_adress,s_n,version,name,D,reso_long,reso_lat,t,dim_bande,dim_gauss,h_range,r_step,phi_rot,r_eff*10**6,\
          stud,domain)
    else :
        s_m = "%sI_%s_%.1f_%s_%i_%i%i_%i_%i_%s_%i_%.2f_%.2f_%s_%s.npy"\
        %(save_adress,s_n,version,name,D,reso_long,reso_lat,t,dim_bande,h_range,r_step,phi_rot,r_eff*10**6,\
          stud,domain)

    return s_m

########################################################################################################################

def index_active (n_species,n_species_cross,n_species_active):

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


def sort_set_param(P,T,Q,gen,compo,Marker=False,Clouds=False,Composition=False) :

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

    if Composition == True :

        sh2 = np.shape(compo)
        compo_rmd = np.zeros((sh2[0],sh[0]*sh[1]*sh[2]))

    else :

        compo_r = 0

    for i in range(sh[0]) :

        for j in range(sh[1]) :

            wh, = np.where((P[i,j,:] != 0))

            P_rmd[ind:ind+wh.size] = P[i,j,wh]
            T_rmd[ind:ind+wh.size] = T[i,j,wh]

            if Marker == True :

                Q_rmd[ind:ind+wh.size] = Q[i,j,wh]

            if Clouds == True :

                gen_rmd[:,ind:ind+wh.size] = gen[:,i,j,wh]

            if Composition == True :

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

    if Marker == True :

        Q_rmd = Q_rmd[wh]
        Q_s = Q_rmd[indices]
        del Q_rmd

    if Clouds == True :

        gen_rmd = gen_rmd[:,wh]
        gen_s = gen_rmd[:,indices]
        del gen_rmd

    if Composition == True :

        compo_rmd = compo_rmd[:,wh]
        compo_s = compo_rmd[:,indices]
        del compo_rmd

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

    if Marker == True :

        Q_r = np.delete(Q_s,list)

    if Clouds == True :

        gen_r = np.delete(gen_s,list,axis=1)

    if Composition == True :

        compo_r = np.delete(compo_s,list,axis=1)

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

    I = np.zeros((dim_bande,reso_alt,reso_theta))

    for i in range(number) :

        step = reso_cut*(i+1)

        I1 = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR%i.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type,step))

        I[i*reso_cut:step] = I1

        if Remove == True :

            os.remove('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR%i.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type,step))

    np.save('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,type),I)


########################################################################################################################


def mixing_contribution_HR(adress,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,c_name,\
                           Isolated,Continuum,Scattering,Clouds,single) :

    stud = stud_type(single,Continuum,Isolated,Scattering,Clouds)

    t_corr = 0
    t_cont = 0
    t_sca = 0
    t_cloud = 0

    if Isolated == False :

        I_corr = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_nude_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_corr = np.log(I_corr)

    if Continuum == True :

        I_cont = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cont_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_cont = np.log(I_cont)

    if Scattering == True :

        I_sca = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_sca_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))
        t_sca = np.log(I_sca)

    if Clouds == True :

        if single == 'no' :

            I_cloud = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cloud_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff))

        else :

            I_cloud = np.load('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_cloud_%s_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,c_name[single]))

        t_cloud = np.log(I_cloud)

    tau = t_corr + t_cont + t_sca + t_cloud

    I = np.exp(tau)

    np.save('%s%s/I_%s_%s_3_%i%i_5_%i_%i_%i_%.2f_%.2f_%s_HR.npy'%(adress,name,version,name,reso_long,reso_lat,dim_bande,lim_alt,reso_step,phi_rot,reff,stud),I)


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


def atmosphere_plot(I_tot,R,h,param,factor,theta_step,r_step,theta_number,i_bande,bande_sample,name,Rp,R_s,extra,trans) :

    dim = int((R+1.5*h)/param)
    from pytransfert import atmospectre

    I_png = np.ones((2*dim,2*dim))
    I = I_tot[i_bande,:,:]

    bar = ProgressBar(2*dim+1,'Image generation')

    for i in range(-dim,dim) :

        find_d = 0
        find_u = 0

        for j in range(-dim,dim) :

            rho = np.sqrt((i*param)**2 + (j*param)**2)

            if rho <= R + h :

                if rho >= R :

                    theta = -math.atan2(i*param,j*param)

                    if theta >= 0 :

                        theta_line = int(round(theta/theta_step))
                        #print(i,j,theta,theta_line)

                    else :

                        theta_line = theta_number + int(round(theta/theta_step))
                        #print(i,j,theta,theta_line)

                        if theta_line == theta_number :

                            theta_line = 0

                    r = rho - R
                    r_line = int(round(r/r_step))

                    I_png[i+dim,j+dim] = I[r_line,theta_line]

                    #if r_line == 0 :

                        #I_png[i+dim,j+dim] = 1.
                else :

                    I_png[i+dim,j+dim] = 'nan'

        bar.animate(i+dim+1)

    #np.save("%s.npy"%(name),I_png)

    x = np.arange(-dim*param,dim*param,param)
    y = -np.arange(-dim*param,dim*param,param)
    X,Y = np.meshgrid(x,y)
    Z = I_png
    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar = atmospectre(I_tot,bande_sample,R_s,Rp,r_step*factor,extra,trans,True)
    R_eff = (R_eff_bar[i_bande*2]/factor - Rp/factor + R)
    print(R_eff_bar[i_bande*2]/factor,Rp/factor,R_eff)

    plt.imshow(I_png, extent = [-dim*param*factor/1000.,dim*param*factor/1000.,-dim*param*factor/1000.,dim*param*factor/1000.])
    plt.colorbar()
    lev = np.array([0,np.exp(-1),0.73,0.99])
    CS = plt.contour(X*factor/1000.,Y*factor/1000.,Z,levels=lev,colors='k')
    plt.clabel(CS,frontsize = 3, inline = 0)
    plt.plot(x*factor/1000.,np.sqrt(((R_eff)**2 - x**2))*factor/1000.,'--m',linewidth = 3)
    plt.plot(x*factor/1000.,-np.sqrt(((R_eff)**2 - x**2))*factor/1000.,'--m', linewidth = 3)
    plt.plot(x*factor/1000.,np.sqrt(((R+h)**2 - x**2))*factor/1000.,'--k',linewidth = 3)
    plt.plot(x*factor/1000.,-np.sqrt(((R+h)**2 - x**2))*factor/1000.,'--k', linewidth = 3)
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

                R_eff_bar[:,theta],R_eff[:,theta],ratio_bar,ratR_bar,bande_bar = atmospectre(I_theta,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=True)

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

                R_eff_bar,R_eff[:,theta],ratio_bar,ratR_bar,bande_bar = atmospectre(I_theta,bande_sample,R_s,Rp,r_step,extra,trans=False,Kcorr=False)

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
