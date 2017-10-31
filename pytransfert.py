import numpy as np
import matplotlib.pyplot as plt
import math as math
import scipy.interpolate as interpld
from pyremind import *
from pygcmtreat import *
from pykcorr import *
from pyfunction import *
from pydataread import *
from pyconstant import *
from netCDF4 import *
from netCDF4 import Dataset
import math as math
import os,sys
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
number_rank = comm.size

########################################################################################################################
########################################################################################################################

"""
    PYTRANSFERT

    Cette bibliotheque intervient a deux niveaux : premierement lors dela resolution du transfert radiatif en
    transmission, puis dans la generation des spectres d'absorption ou des courbes de lumiere.

    Les fonctions ci-dessous appellent des fichiers de donnee precalcules notamment par les routines de pkcorr. Elles
    exploitent egalement l'optimisation apportee par pyremind qui allege considerablement le nombre de calculs et par
    consequent, les temps de calcul.

    La deuxieme partie des fonctions ici definies servent a interpreter les cartes de transmittances produites par les
    fonction trans2fert. Seule la transmission est ici prise en compte, un autre module aura pour responsabilite d'ajouter
    l'emission a ces cartes

    Version : 6.0

    Recentes mises a jour :

    >> Suppression du retour de N dans altitude_line_array1D_cyl_optimized_correspondance
    >> Suppression de l'entree T dans altitude_line_array1D_cyl_optimized_correspondance
    >> Modification d'atmospectre, desormais si data_convert a ete calcule des le depart sur la base des milieux de
    couche le spectre est calcule sur cette base, les aires associees sont donc plus realistes.
    >> Correction sur les equations d'effective_radius qui ne calculait pas correctement les aires de recouvrement
    >> Suppression de M_air et g dans les parametres d'entree des routines trans2fert1D et trans2fert3D

    Date de derniere modification : 10.10.2016

    >> Refonte des fonctions de transfert radiatif et introduction de la possibilite d'integrer la profondeur optique
    >> sur le chemin optique des rayons. Prise en compte des differentes interpolations possibles par rapport a la
    >> temperature dans convertator1D
    >> Possibilite dans le cas discret de moduler la densite de reference grace a la fonction Module
    >> Fini pour les effets de bord et les effets aux poles
    >> Verification d'atmospectre et possibilite de soumettre des grilles de transmittance sur un point theta ou sur une
    >> bande unique

    Date de derniere modification : 12.12.2016

"""

########################################################################################################################
########################################################################################################################

"""
    TRANS2FERT1D

    Cette fonction exploite l'ensemble des outils developpes precedemment afin de produire une carte de transmittance
    dans une maille cylindrique. Si la maille n'est necessaire en soit, une seule longitude est utile dans la production
    des transits ou dans les estimations du rayon effectif (au premier ordre). Cette routine peut effectuer une
    interpolation sur les donnees, toutefois le temps de calcul est tres nettement augmente (plusieurs dizaines d'heure)
    L'utilisation des donnees brutes peut etre traite en utilisant ou non les tables dx_grid et order_grid pre-etablie.
    En fonction de la resolution initiale adoptee pour les donnees GCM, les tables dx et order permettent un petit gain
    de temps (pour les resolutions elevees).

    La production de la colonne peut etre effectuee en amont eventuellement.

    Cette fonction retourne la grille de transmittance dans une maille cylindrique I[bande,r,theta].

"""

########################################################################################################################
########################################################################################################################


def trans2fert1D (k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,T_cont,Q_cloud,Rp,h,g0,r_step,theta_step,\
                  x_step,gauss,gauss_val,dim_bande,data,P_col,T_col,gen_col,Q_col,compo_col,ind_active,dx_grid,order_grid,pdx_grid,\
                  P_sample,T_sample,Q_sample,bande_sample,name_file,n_species,c_species,single,\
                  bande_cloud,r_eff,r_cloud,rho_p,t,phi_rot,domain,ratio,lim_alt,rupt_alt,directory,z_grid,type,\
                  Marker=False,Continuum=True,Isolated=False,Scattering=True,Clouds=False,Kcorr=True,Rupt=False,\
                  Middle=False,Integral=False,Module=False,Optimal=False,D3Maille=False) :

    r_size,theta_size,x_size = np.shape(dx_grid)
    number_size,z_size,lat_size,long_size = np.shape(data)

    Composition = True

    if Kcorr == True :

        k_rmd = np.zeros((T_col.size,dim_bande,gauss.size))

    else :

        k_rmd = np.zeros((T_col.size,dim_bande))

    Itot = np.ones((dim_bande,r_size,1))

    bar = ProgressBar(r_size,'Radiative transfert progression')

    for i in range(1) :

        if i == 0 :

            theta_line = i

            if Rupt == True :

                dep = int(rupt_alt/r_step)

                Itot[:, 0:dep, theta_line] = np.zeros((dim_bande,dep))

            else :

                dep = 0

            for j in range(dep,r_size) :

                if Middle == False :

                    r = Rp + j*r_step

                else :

                    r = Rp + (j+0.5)*r_step
                r_line = j

                dx = dx_grid[r_line,theta_line,:]
                order = order_grid[:,r_line,theta_line,:]
                pdx = pdx_grid[r_line,theta_line,:]

                if r < Rp + lim_alt :

                    if j == 0 :

                        P_rmd,T_rmd,Q_rmd,k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd = \
                        convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo_col,ind_active,\
                        k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,Q_cloud,T_cont,P_sample,\
                        T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name_file,t,phi_rot,\
                        n_species,domain,ratio,directory,Marker,Composition,Continuum,Scattering,Clouds,Kcorr,Optimal)

                    zone, = np.where(dx >= 0)
                    cut, = np.where(order[0,zone] < z_size)
                    dx_ref = dx[zone[cut]]
                    #data_ref = data[:,order[0,zone[cut]],order[1,zone[cut]],order[2,zone[cut]]]
                    data_ref = data[:,order[0,zone[cut]],0,0]
                    P_ref, T_ref = data_ref[0], data_ref[1]

                    if Marker == True :

                        Q_ref = data_ref[2]

                        k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                        k_correlated_interp_remind_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Q_rmd,Q_ref,Continuum,Scattering,Clouds,Kcorr)

                    else :

                        k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                        k_correlated_interp_remind(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Continuum,Scattering,Clouds,Kcorr)

                    if Module == True :
                        z_ref = z_grid[r_line,theta_line,order[3,zone[cut]]]
                        P_ref = module_density(P_ref,T_ref,z_ref,Rp,g0,data_ref[number_size-1],r_step,type,Middle)
                    Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

                    I_out = radiative_transfert_remind3D(dx_ref,pdx[zone[cut]],Cn_mol_ref,k_inter,k_cont_inter,k_sca_inter,\
                            k_cloud_inter,gauss_val,single,Continuum,Isolated,Scattering,Clouds,Kcorr,Integral)

                    Itot[:, r_line, 0] = I_out[:]

                bar.animate(j+1)

    return Itot


########################################################################################################################
########################################################################################################################

"""
    TRANS2FERT3D

    Cette fonction exploite l'ensemble des outils developpes precedemment afin de produire une carte de transmittance
    dans une maille cylindrique. Cette routine peut effectuer une interpolation sur les donnees, toutefois le temps de
    calcul est tres nettement augmente (plusieurs dizaines d'heure)
    L'utilisation des donnees brutes peut etre traite en utilisant ou non les tables dx_grid et order_grid pre-etablie.
    En fonction de la resolution initiale adoptee pour les donnees GCM, les tables dx et order permettent un petit gain
    de temps (pour les resolutions elevees).

    La production de la colonne peut etre effectuee en amont eventuellement.

    Cette fonction retourne la grille de transmittance dans une maille cylindrique I[bande,r,theta].

"""

########################################################################################################################
########################################################################################################################


def trans2fert3D (k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,Rp,h,g0,r_step,theta_step,gauss_val,dim_bande,data,\
                  P_rmd,T_rmd,Q_rmd,dx_grid,order_grid,pdx_grid,z_grid,t,\
                  name_file,n_species,single,rmind,lim_alt,rupt_alt,rank,rank_ref,\
                  Marker=False,Continuum=True,Isolated=False,Scattering=True,Clouds=True,Kcorr=True,\
                  Rupt=False,Module=False,Integral=False,TimeSel=False) :

    r_size,theta_size,x_size = np.shape(dx_grid)
    number_size,t_size,z_size,lat_size,long_size = np.shape(data)

    if TimeSel == True :
        data = data[:,t,:,:,:]

    Itot = np.ones((dim_bande,r_size,theta_size))

    if rank == rank_ref : 
        bar = ProgressBar(int(round(2*np.pi/theta_step)),'Radiative transfert progression')

    for i in range(theta_size) :

        theta_line = i
        fail = 0

        if Rupt == True :

            dep = int(rupt_alt/r_step)

            Itot[:, 0:dep, theta_line] = np.zeros((dim_bande,dep))

        else :

            dep = 0

        for j in range(dep,r_size) :

            r = Rp + j*r_step
            r_line = j

            dx = dx_grid[r_line,theta_line,:]
            order = order_grid[:,r_line,theta_line,:]
            if Integral == True :
                pdx = pdx_grid[r_line,theta_line,:]

            if r < Rp + lim_alt :

                zone, = np.where((order[0] > 0)*(order[0] < z_size))
                dx_ref = dx[zone]
                if Integral == True :
                    pdx_ref = pdx[zone]
                else :
                    pdx_ref = np.array([])
                data_ref = data[:,order[0,zone],order[1,zone],order[2,zone]]
                P_ref, T_ref = data_ref[0], data_ref[1]

                if Marker == True :

                    Q_ref = data_ref[2]

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail = \
                    k_correlated_interp_remind3D_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,P_ref.size,\
                    P_rmd,P_ref,T_rmd,T_ref,Q_rmd,Q_ref,n_species,fail,rmind,Continuum,Isolated,Scattering,Clouds,Kcorr)

                else :

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail = \
                    k_correlated_interp_remind3D(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,P_ref.size,\
                    P_rmd,P_ref,T_rmd,T_ref,n_species,fail,rmind,Continuum,Isolated,Scattering,Clouds,Kcorr)

                if Module == True :
                    z_ref = z_grid[r_line,theta_line,order[3,zone]]
                    P_ref = module_density(P_ref,T_ref,z_ref,Rp,g0,data_ref[number_size-1],r_step,type,True)
                Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

                I_out = radiative_transfert_remind3D(dx_ref,pdx_ref,Cn_mol_ref,k_inter,k_cont_inter,k_sca_inter,\
                    k_cloud_inter,gauss_val,single,Continuum,Isolated,Scattering,Clouds,Kcorr,Integral)

                Itot[:, r_line, theta_line] = I_out

        if rank == rank_ref : 
	    bar.animate(i+1)

        if fail !=0 and rank == rank_ref :

            print("%i failure(s) at this latitude" %(fail))

    return Itot


########################################################################################################################
########################################################################################################################

"""
    EFFECTIVE_RADIUS

    Dans le cas d'une etoile uniforme en luminosite et parfaitement spherique, nous cherchons a determiner le rayon
    effectif d'une exoplanete transitant devant l'etoile. Nous integrons alors la grille cylindrique de transmittance
    et nous la soustrayons a la luminosite totale normalisee de l'etoile. Nous considerons pour cela une intensite de
    1 par metre carre.

    Cette fonction retourne le rayon effectif ainsi que l'epaisseur de l'atmosphere afin de verifier la pertinence du
    resultat.

"""

########################################################################################################################
########################################################################################################################


def effective_radius(I,R_s,Rp,r_step,extra,Middle=False) :

    I_p = 0.
    I_p2 = 0.
    r_size,theta_size = np.shape(I)
    A_surf = np.pi*Rp**2
    A_surf_arr = np.zeros((r_size,theta_size),dtype='float')
    I_sol = np.pi*R_s**2
    R = Rp

    for r in range(r_size) :

        if r == 0 :

            if Middle == False :
                A = np.pi*(1/2.)*(2.*R + 1/2.*r_step)*r_step/float(theta_size)
                R += r_step/2.
            else :
                A = 2.*np.pi*r_step*((r+0.5)*r_step + Rp)

            A_surf += A
            A_surf_arr[r,:] = np.ones(theta_size,dtype='float')*A/np.float(theta_size)

        else :

            if Middle == False :
                A = np.pi*(2*R + r_step)*r_step/float(theta_size)
                R += r_step
            else :
                A = 2.*np.pi*r_step*((r+0.5)*r_step + Rp)

            A_surf += A
            A_surf_arr[r,:] = np.ones(theta_size,dtype='float')*A/np.float(theta_size)

        for theta in range(theta_size) :

            I_p += I[r,theta]*A/np.float(theta_size)
            I_p2 += (1-I[r,theta])*A/np.float(theta_size)

    I_tot = I_sol - A_surf + I_p

    R_p = R_s*np.sqrt((I_sol - I_tot)/(I_sol))

    h_eff = R_p - Rp

    return R_p, h_eff, A_surf, A_surf_arr, I_p2


########################################################################################################################


def effective_radius_boucle(I,R_s,Rp,r_step,A_surf,A_surf_arr,extra) :

    I_sol = np.pi*R_s**2

    I_p = np.nansum(I*A_surf_arr)

    I_tot = I_sol - A_surf + I_p

    h_eff = - Rp +np.sqrt(A_surf/(np.pi)) - extra*r_step

    R_p = R_s*np.sqrt((I_sol - I_tot)/(I_sol))

    return R_p, h_eff


########################################################################################################################


def atmospectre(I,bande_sample,R_s,Rp,r_step,extra,trans,Kcorr=False,Middle=False) :

    if Kcorr == True :

        bande = np.array([])
        bande_bar = np.array([])

        dim_bande = bande_sample.size - 1

        for i in range(dim_bande+1) :

            if i == 0 :

                bande = np.append(bande, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))

            else :

                bande = np.append(bande, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))

        bande_bar = np.delete(bande_bar,bande_bar.size-1)

        R_eff = np.array([])
        R_eff_bar = np.array([])
        ratio = np.array([])
        ratio_bar = np.array([])

        for i in range(dim_bande) :

            if i == 0 :

                if trans == False :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(I[i,:,:],R_s,Rp,r_step,extra,Middle)
                else :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(np.transpose(I[i,:,:]),R_s,Rp,r_step,extra,Middle)

            else :

                if trans == False :
                    R,h = effective_radius_boucle(I[i,:,:],R_s,Rp,r_step,A_surf,A_surf_arr,extra)
                else :
                    R,h = effective_radius_boucle(np.transpose(I[i,:,:]),R_s,Rp,r_step,A_surf,A_surf_arr,extra)

            R_eff = np.append(R_eff, np.array([R]))
            R_eff_bar = np.append(R_eff_bar, np.array([R]))
            R_eff_bar = np.append(R_eff_bar, np.array([R]))

        R_ref = np.amin(R_eff)

        ratio_bar = np.append(ratio_bar, (R_eff_bar - R_ref)/R_ref*1000000.)
        ratR_bar = (R_eff_bar - R_ref)/R_s*1000000.

        flux_bar = R_eff_bar**2/R_s**2
        flux = 0

    else :

        dim_bande = bande_sample.size

        R_eff = np.array([])

        for i in range(dim_bande) :

            if i == 0 :

                if trans == False :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(I[i,:,:],R_s,Rp,r_step,extra,Middle)
                else :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(np.transpose(I[i,:,:]),R_s,Rp,r_step,extra,Middle)

            else :

                if trans == False :
                    R,h = effective_radius_boucle(I[i,:,:],R_s,Rp,r_step,A_surf,A_surf_arr,extra)
                else :
                    R,h = effective_radius_boucle(np.transpose(I[i,:,:]),R_s,Rp,r_step,A_surf,A_surf_arr,extra)

            R_eff = np.append(R_eff, np.array([R]))

        R_eff_bar = 0
        ratio_bar = 0
        ratR_bar = 0
        bande_bar = 0

        flux = R_eff**2/R_s**2
        flux_bar = 0

    return R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux


########################################################################################################################


def stellarint(Rs,U,param,Law) :

    Is = 0

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0]*(1 - np.cos(q)) - U[1]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(q)) - U[1]*(1 - q) - U[2]*(1 - q**(3/2.)) - U[3]*(1 - q**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is += I2(d,U)*np.pi*((d*Rs+param)**2 - (d*Rs)**2)

    return Is


########################################################################################################################


def stellarint_matrix (Rs,U,param,Law) :

    dim_coeff,dim_ban = np.shape(U)
    Is = np.zeros(dim_ban)

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0,:]*(1 - np.cos(q)) - U[1,:]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0,:]*(1 - np.sqrt(q)) - U[1,:]*(1 - q) - U[2,:]*(1 - q**(3/2.)) - U[3,:]*(1 - q**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is += I2(d,U)*np.pi*((d*Rs+param)**2 - (d*Rs)**2)

    return Is

########################################################################################################################


def stellarplot (Rs,U,param,Law) :

    width = int(Rs/param)
    Is = np.zeros(width)

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0]*(1 - np.cos(q)) - U[1]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(np.cos(q))) - U[1]*(1 - np.cos(q)) - U[2]*(1 - np.cos(q)**(3/2.)) - U[3]*(1 - np.cos(q)**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is[i] = I2(d,U)

    Is_png = np.zeros((2*width+1,2*width+1))

    for i in range(width+1) :

        for j in range(width+1) :

            rho = np.sqrt(i**2 + j**2)
            index = int(np.round(rho))

            if index < width :

                Is_png[width+i,width+j] = Is[index]
                Is_png[width-i,width+j] = Is[index]
                Is_png[width+i,width-j] = Is[index]
                Is_png[width-i,width-j] = Is[index]

        if i%10 == 0 :

            print(i)

    print(Is)

    plt.imshow(Is_png,aspect='auto',extent=[-width*param,width*param,width*param,-width*param])
    plt.colorbar()
    plt.show()

    return Is, Is_png



########################################################################################################################


def lightcurve_maker(Rs,Rp,h,Ms,cDcorr,cPcorr,inc_0,U_limb,Itot_plan,r_step,theta_step,reso_occ,t,i_sample,bande_sample,bande_limb,Law) :

    reso_bande,reso_r,reso_theta = np.shape(Itot_plan)

    I_time = np.ones((i_sample.size,t.size))
    I_ground_time = np.zeros((i_sample.size,t.size))
    I_ground_plan_time = np.zeros((i_sample.size,t.size))
    I_star_time = np.ones((i_sample.size,t.size))

    a = ((((cPcorr*24*3600)**2)*G*(Ms/(4*np.pi**2)))**(1/3.))/(Rs)
    r = (Rp + h)/Rs

    ksi = 2*np.pi*(t - cDcorr)/cPcorr
    u = - np.cos(inc_0)*np.cos(ksi)
    v = np.sin(ksi)
    sep = a*np.sqrt(u**2 + v**2)


    #B2 = lambda q,ld1,ld2,sep,r : 2*q*I2(q,ld1,ld2)*np.arccos((q**2 + sep**2 - r**2)/(2*sep*q))
    #Fs2 = np.pi*(1 - ld1/3. - ld2/2.)

    for i_bande in range(i_sample.size) :

        Itot = Itot_plan[i_sample[i_bande],:,:]
        wave = 1./(bande_sample[i_sample[i_bande]]*100)*10**6
        wh, = np.where(bande_limb >= wave)

        if wh.size != 0 :

            index = wh[0] - 1

        else :

            index = bande_limb.size - 2

        U = U_limb[:,index]
        Is = stellarint(Rs,U,1000,"Nonlinear")

        print('Total stellar flux = %s' %(Is))

        if Law == "Quadratic" :

            #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
            I2 = lambda q,U : 1 - U[0]*(1 - q) - U[1]*(1 - q)**2

        if Law == "Nonlinear" :

            I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(q)) - U[1]*(1 - q) - U[2]*(1 - q**(3/2.)) - U[3]*(1 - q**2)

        gre, = np.where(sep < 1 + r)

        for i in gre :

            I_ground = np.zeros((reso_r,reso_theta))
            I_ground_plan = np.zeros((reso_occ,reso_theta))

            u_xy = - np.cos(inc_0)*np.cos(ksi[i])
            v_xy = np.sin(ksi[i])

            sep_xy = a*np.sqrt(u_xy**2 + v_xy**2)

            if sep_xy < 1 - r :

                for r_range in range(reso_r) :

                    for theta_range in range(reso_theta) :

                        y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                        x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                        pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                        #print(sep_xy,r,x,y)

                        rho = np.sqrt((sep_xy + x)**2 + y**2)
                        I_ground[r_range,theta_range] = I2(np.cos(rho),U)*pond
                        #print(I_ground[j,:])

                for r_range in range(1,reso_occ) :

                    for theta_range in range(reso_theta) :

                        y = (r_range*r/(Rs*float(reso_occ)))*np.sin(theta_range*theta_step)
                        x = (r_range*r/(Rs*float(reso_occ)))*np.cos(theta_range*theta_step)
                        pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                        #print(sep_xy,r,x,y)

                        rho = np.sqrt((sep_xy + x)**2 + y**2)
                        I_ground_plan[r_range,theta_range] = I2(np.cos(rho),U)*pond

                print("first")

            if sep_xy >= 1 - r and sep_xy <= 1 + r :

                for r_range in range(reso_r) :

                    for theta_range in range(reso_theta) :

                        y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                        x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                        pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                        rho = np.sqrt((sep_xy + x)**2 + y**2)

                        #print(r_range,theta_range,rho)

                        if rho <= 1 :

                            #print(ok)
                            I_ground[r_range,theta_range] = I2(np.cos(rho),U)*pond
                            #print(I_ground[j,:])

                for r_range in range(1,reso_occ) :

                    for theta_range in range(reso_theta) :

                        y = (r_range*r/(float(reso_occ)))*np.sin(theta_range*theta_step)
                        x = (r_range*r/(float(reso_occ)))*np.cos(theta_range*theta_step)
                        pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                        rho = np.sqrt((sep_xy + x)**2 + y**2)

                        #print(r_range,theta_range,rho)

                        if rho <= 1 :

                            I_ground_plan[r_range,theta_range] = I2(np.cos(rho),U)*pond

                print('second %s'%(sep_xy),r)

            Itot_result = Itot*I_ground
            #plt.imshow(I_ground, aspect='auto')
            #plt.colorbar()
            #plt.show()
            #plt.imshow(I_ground_plan, aspect='auto')
            #plt.colorbar()
            #plt.show()
            I_time[i_bande,i] = np.nansum(Itot_result)
            I_ground_time[i_bande,i] = np.nansum(I_ground)
            I_ground_plan_time[i_bande,i] = np.nansum(I_ground_plan)

            I_star_time[i_bande,i] = 1 - I_ground_time[i_bande,i]/Is - I_ground_plan_time[i_bande,i]/Is + I_time[i_bande,i]/Is
            print(I_star_time[i_bande,i])

        les, = np.where(sep > 1 + r)

        for i in les :

            I_star_time[i_bande,i] = 1

        #plt.imshow(I_ground,vmin=0.52,vmax=0.57)
        #plt.colorbar()
        #plt.show()

    return I_star_time


########################################################################################################################


def lightcurve_maker_opt(Rs,Rp,h,Ms,cDcorr,cPcorr,inc_0,U_limb,Itot_plan,r_step,theta_step,reso_occ,t,i_sample,bande_sample,bande_limb,Law) :

    reso_bande,reso_r,reso_theta = np.shape(Itot_plan)
    Itot = Itot_plan[i_sample,:,:]
    dim_coeff,dim_ban = np.shape(U_limb)

    I_time = np.ones((i_sample.size,t.size))
    I_ground_time = np.zeros((i_sample.size,t.size))
    I_ground_plan_time = np.zeros((i_sample.size,t.size))
    I_star_time = np.ones((i_sample.size,t.size))

    a = ((((cPcorr*24*3600)**2)*G*(Ms/(4*np.pi**2)))**(1/3.))/(Rs)
    r = (Rp + h)/Rs

    ksi = 2*np.pi*(t - cDcorr)/cPcorr
    u = - np.cos(inc_0)*np.cos(ksi)
    v = np.sin(ksi)
    sep = a*np.sqrt(u**2 + v**2)

    U = np.zeros((dim_coeff,i_sample.size))

    for i_bande in range(i_sample.size) :

        wave = 1./(bande_sample[i_sample[i_bande]]*100)*10**6
        wh, = np.where(bande_limb >= wave)

        if wh.size != 0 :

            index = wh[0] - 1

        else :

            index = bande_limb.size - 2

        U[:,i_bande] = U_limb[:,index]

    Is = stellarint_matrix(Rs,U,1000,"Nonlinear")

    print('Total stellar flux = %s' %(Is))

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0,:]*(1 - q) - U[1,:]*(1 - q)**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0,:]*(1 - np.sqrt(q)) - U[1,:]*(1 - q) - U[2,:]*(1 - q**(3/2.)) - U[3,:]*(1 - q**2)

    gre, = np.where(sep < 1 + r)

    for i in gre :

        I_ground = np.zeros((i_sample.size,reso_r,reso_theta))
        I_ground_plan = np.zeros((i_sample.size,reso_occ,reso_theta))

        u_xy = - np.cos(inc_0)*np.cos(ksi[i])
        v_xy = np.sin(ksi[i])

        sep_xy = a*np.sqrt(u_xy**2 + v_xy**2)

        if sep_xy < 1 - r :

            for r_range in range(reso_r) :

                for theta_range in range(reso_theta) :

                    y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                    x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                    pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                    #print(sep_xy,r,x,y)

                    rho = np.sqrt((sep_xy + x)**2 + y**2)
                    I_ground[:,r_range,theta_range] = I2(np.cos(rho),U)*pond
                    #print(I_ground[j,:])

            for r_range in range(1,reso_occ) :

                for theta_range in range(reso_theta) :

                    y = (r_range*r/(Rs*float(reso_occ)))*np.sin(theta_range*theta_step)
                    x = (r_range*r/(Rs*float(reso_occ)))*np.cos(theta_range*theta_step)
                    pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                    #print(sep_xy,r,x,y)

                    rho = np.sqrt((sep_xy + x)**2 + y**2)
                    I_ground_plan[:,r_range,theta_range] = I2(np.cos(rho),U)*pond

            print("first")

        if sep_xy >= 1 - r and sep_xy <= 1 + r :

            for r_range in range(reso_r) :

                for theta_range in range(reso_theta) :

                    y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                    x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                    pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                    rho = np.sqrt((sep_xy + x)**2 + y**2)

                    #print(r_range,theta_range,rho)

                    if rho <= 1 :

                        #print(ok)
                        I_ground[:,r_range,theta_range] = I2(np.cos(rho),U)*pond
                        #print(I_ground[j,:])

            for r_range in range(1,reso_occ) :

                for theta_range in range(reso_theta) :

                    y = (r_range*r/(float(reso_occ)))*np.sin(theta_range*theta_step)
                    x = (r_range*r/(float(reso_occ)))*np.cos(theta_range*theta_step)
                    pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                    rho = np.sqrt((sep_xy + x)**2 + y**2)

                    #print(r_range,theta_range,rho)

                    if rho <= 1 :

                        I_ground_plan[:,r_range,theta_range] = I2(np.cos(rho),U)*pond

            print('second %s'%(sep_xy),r)

        Itot_result = Itot*I_ground

        for i_bande in range(i_sample.size) :

            I_time[i_bande,i] = np.nansum(Itot_result[i_bande,:,:])
            I_ground_time[i_bande,i] = np.nansum(I_ground[i_bande,:,:])
            I_ground_plan_time[i_bande,i] = np.nansum(I_ground_plan[i_bande,:,:])

            I_star_time[i_bande,i] = 1 - I_ground_time[i_bande,i]/Is[i_bande] - I_ground_plan_time[i_bande,i]/Is[i_bande]\
                        + I_time[i_bande,i]/Is[i_bande]

    les, = np.where(sep > 1 + r)

    for i in les :

        I_star_time[:,i] = 1

    return I_star_time
