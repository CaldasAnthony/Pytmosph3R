from pyfunction import *
from pyconstant import *
from netCDF4 import Dataset
import math as math
import scipy.integrate as integrate
import os,sys
import time

########################################################################################################################
########################################################################################################################

"""
    PYGCMTREAT

    Cette bibliotheque intervient a deux niveaux : premierement lors de la generation des matrices de conversion qui
    permettent de transposer les donnes a symetrie spherique du GCM dans la maille cylindrique utilisee par pytransfert.
    Deuxiemement dans la recuperation des trajets de rayons dans la dite maille et la recuperation des donnees pre-
    calculees.

    La fonction Boxes effectue le meme travail que zrecast du GCM, nous pouvons neanmoins aisement modifier l'echelle en
    altitude et extrapoler cette derniere pour la haute atmosphere, notamment lorsque le toit du modele est trop bas par
    rapport aux proprietes absorbantes en transmission de son atmosphere. Les fonctions ci-dessous tiennent compte de la
    presence de marqueurs, de nuages, du nombre de molecule d'interet, et de leurs influences sur le profil atmospherique.

    Nous pouvons ici tenir compte de la rotation de l'exoplanete ou d'une eventuelle obliquite.

    Version : 6.0

    Recentes mises a jour :

    >> Modifications de altitude_line_array1D_cyl_optimized_correspondance suite a une mauvaise estimation du l qui
    calculait l'epaisseur de correction sur les bords de l'atmosphere
    >> Suppression du retour de N
    >> Suppression de l'entree T
    >> Memes modifications altitude_line_array2D_cyl_optimized_correspondance
    >> cylindric_assymatrix_parameter va desormais pouvoir effectuer les chemins optiques au centre des couches de boxes
    a la limite inferieure ou a la limite superieure. Le nombre de rayon sortant est donc different et l'association des
    parametres necessite une nouvelle interpolation (modification necessaire de convertator puisque la diversite de P, T
    Q va necessairement varier de data_convert
    >> Nouvelle modification de Boxes, desormais la fonction peut calculer preferentiellement les proprietes de l'
    atmosphere au milieu des couches regulieres de la grille en altitude qui est souhaite. Ainsi, data_convert[:,:,0,:,:]
    correspond aux parametres de surface, et data_convert[:,:,1:,:,:] aux parametres au milieu des couches
    >> cylindric_assymatrix_parameter est donc a nouveau modifiee de maniere a tenir compte du fait que les data soient
    deja determines au milieu des couches, ainsi si l'option Middle est choisie, l'ordre de z_array devient l'ordre de
    la couche, et on conserve bien les proprietes, nous n'avons plus de demi-couche pour la surface puisque les
    parametres de surface n'interviennent plus (nous gardons cet ordre 0 de surface pour garder les informations sur
    le diagfi de la simu GCM).
    Note : le z_array doit rester un tableau allant de 0 a h avec un pas de delta_z, il donne la position des inter-
    couches
    >> Modification de dx_correspondance qui desormais calcule exactement les distances parcourues par un rayon dans
    les differentes cellules de l'atmosphere a une couche et un point de lattitude donnee. La fonction peut faire l'
    objet d'une amelioration en tenant compte des symetrie sur ces distances. On notera que les cas particuliers ou non
    seulement la strate en altitude et le point de latiude ou de longitude changent, les calculs favorisent le saut de
    couche sur les sauts de latitude ou longitude
    >> Correction d'un bug sur le calcul des l sur les chemins optiques, cette grandeur peut etre positive (il manque
    une partie de cellule) ou negative (la derniere cellule depasse le toit de l'atmosphere)

    Date de derniere modification : 14.09.2016

    >> Modification de dx_correspondance, les distances calculees sont desormais plus precises que jamais et decoupent
    reellement le chemin optique en deux pour s'abstenir des problemes aux poles
    >> Cette meme fonction peut desormais integrer sur le chemin optique des rayons la profondeur optique (divisee de
    la section efficace toutefois), nous ne tenons pas compte d'une eventuelle dependance de la fraction molaire ou de
    la section efficace avec l'altitude. Pour que cette hypothese reste valable, le pas en altitude doit etre bien
    inferieur a la hauteur d'echelle
    >> Desormais, chaque changement d'altitude, de longitude ou de latitude est traite de maniere independante, on notera
    que l'integrale diverse pour les variations d'altitude tres tres faibles (ce qui arrive typiquement lorsque les
    rayons traversent des coins de cellule spherique ou couramment pour les terminateurs aux poles)
    >> Cette ecriture est bien adaptee au cas Middle, il l'est moins si cette option est False
    >> Bien que conservees, les fonctions altitude_array ne sont plus appelees dans le transfert radiatif
    >> Modification dans la construction du profil P-T cylindrique, la loi hydrostatique a ete reecrite
    >> Une verification serait cependant avisee pour etre certain que cette reecriture ne s'eloigne pas de celle attendue
    (par exemple, cas isotherme)

    Date de derniere modification : 12.12.2016

"""

########################################################################################################################
########################################################################################################################


def Boxes_spheric_data(data,t,c_species,Surf=True,Marker=False,Clouds=False,TimeSelec=False) :

    file = Dataset("%s.nc"%(data))
    variables = file.variables
    c_number = c_species.size

    # Si nous avons l'information sur les parametres de surface

    if Surf == True :

        # Si nous avons l'information sur la pression de surface, il nous faut donc rallonger les tableaux de parametres
        # de 1

        if TimeSelec == False :
            T_file = variables["temp"][:]
            a,b,c,d = np.shape(T_file)
            T_surf = variables["tsurf"][:]
            P_file = variables["p"][:]
            P_surf = variables["ps"][:]
            P = np.zeros((a,b+1,c,d),dtype=np.float64)
            T = np.zeros((a,b+1,c,d),dtype=np.float64)
        else :
            T_prefile = variables["temp"][:]
            a,b,c,d = np.shape(T_prefile)
            T_file = np.zeros((1,b,c,d))
            T_surf = np.zeros((1,c,d))
            P_file = np.zeros((1,b,c,d))
            P_surf = np.zeros((1,c,d))
            T_file[0] = variables["temp"][t,:,:,:]
            T_surf[0] = variables["tsurf"][t,:,:]
            P_file[0] = variables["p"][t,:,:,:]
            P_surf[0] = variables["ps"][t,:,:]
            P = np.zeros((1,b+1,c,d),dtype=np.float64)
            T = np.zeros((1,b+1,c,d),dtype=np.float64)

        P[:,0,:,:] = P_surf
        P[:,1:b+1,:,:] = P_file
        T[:,0,:,:] = T_surf
        T[:,1:b+1,:,:] = T_file

        if Marker == True :
            if TimeSelec == False :
                h2o_vap = variables["h2o_vap"][:]
                h2o_vap_surf = variables["h2o_vap_surf"][:]
                h2o = np.zeros((a,b+1,c,d),dtype=np.float64)

            else :
                h2o_vap = np.zeros((1,b,c,d),dtype=np.float64)
                h2o_vap_surf = np.zeros((1,c,d),dtype=np.float64)
                h2o_vap[0] = variables["h2o_vap"][t,:,:,:]
                h2o_vap_surf[0] = variables["h2o_vap_surf"][t,:,:]
                h2o = np.zeros((1,b+1,c,d),dtype=np.float64)

            h2o[:,0,:,:] = h2o_vap_surf
            h2o[:,1:b+1,:,:] = h2o_vap

        else :
            h2o = np.array([])

        if Clouds == True :
            if TimeSelec == False :
                gen_cond = np.zeros((c_number,a,b,c,d),dtype=np.float64)
                gen_cond_surf = np.zeros((c_number,a,c,d),dtype=np.float64)
                for c_num in range(c_number) :
                    gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][:]
                    gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
                gen = np.zeros((c_species.size,a,b+1,c,d),dtype=np.float64)
            else :
                gen_cond = np.zeros((c_number,1,b,c,d),dtype=np.float64)
                gen_cond_surf = np.zeros((c_number,1,c,d),dtype=np.float64)
                for c_num in range(c_number) :
                    gen_cond_surf[c_num,:,:,:] = variables["%s_surf"%(c_species[c_num])][t,:,:]
                    gen_cond[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
                gen = np.zeros((c_species.size,1,b+1,c,d),dtype=np.float64)

            gen[:,:,0,:,:] = gen_cond_surf
            gen[:,:,1:b+1,:,:] = gen_cond
        else :
            gen = np.array([])

        if TimeSelec == True :
            a = 1
        T_mean = np.nansum(T_file[:,b-1,:,:])/(a*c*d)
        T_max = np.amax(T_file[:,b-1,:,:])
        T_min = np.amin(T_file[:,b-1,:,:])
        print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature : %i K'%(T_mean,T_max,T_min))

        P_mean = np.nansum(P_file[:,b-1,:,:])/(a*c*d)
        print('Mean roof pressure : %f Pa'%(P_mean))

        b = b + 1

    # Si nous n'avons pas l'information sur les parametres de surface

    else :

        if TimeSelec == False :
            T = variables["temp"][:]
            T = np.array(T,dtype = np.float64)
            a,b,c,d = np.shape(T)
            P = variables["p"][:]
            P = np.array(P,dtype = np.float64)
        else :
            T_prefile = variables["temp"][:]
            a,b,c,d = np.shape(T_prefile)
            T = np.zeros((1,b,c,d),dtype=np.float64)
            P = np.zeros((1,b,c,d),dtype=np.float64)
            T[0] = variables["temp"][t,:,:,:]
            P[0] = variables["p"][t,:,:,:]

        if Marker == True :
            if TimeSelec == False :
                h2o = variables["h2o_vap"][:]
                h2o = np.array(h2o,dtype=np.float64)
            else :
                h2o = np.zeros((1,b,c,d),dtype=np.float64)
                h2o[0] = variables["h2o_vap"][t,:,:,:]
        else :
            h2o = np.array([])

        if Clouds == True :
            if TimeSelec == False :
                gen = np.zeros((c_number,a,b,c,d),dtype=np.float64)
                for c_num in range(c_number) :
                    gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][:]
            else :
                gen = np.zeros((c_number,1,b,c,d),dtype=np.float64)
                for c_num in range(c_number) :
                    gen[c_num,:,:,:,:] = variables["%s"%(c_species[c_num])][t,:,:,:]
        else :
            gen = np.array([])

        if TimeSelec == True :
            a = 1
        T_mean = np.nansum(T[:,b-1,:,:]/(a*c*d))
        T_max = np.amax(T[:,b-1,:,:])
        T_min = np.amin(T[:,b-1,:,:])
        print('Mean temperature : %i K, Maximal temperature : %i K, Minimal temperature of the high atmosphere : %i K'\
              %(T_mean,T_max,T_min))

        P_mean = np.nansum(P[:,b-1,:,:])/(a*c*d)
        print('Mean roof pressure : %f Pa'%(P_mean))

    return P, T, h2o, gen


########################################################################################################################
########################################################################################################################


def Boxes_interlopation(P,T,h2o,Rp,g0,M_atm,number,P_comp,T_comp,Q_comp,species,x_species,M_species,c_species,ratio,\
                        Marker=False,Clouds=False,Composition=False,LogInterp=False) :

    a, b, c, d = np.shape(P)
    z = np.zeros((a,b,c,d),dtype=np.float64)
    M = np.zeros((a,b,c,d),dtype=np.float64)

    if Marker == False :

        if Composition == True :

            size = species.size

            compo = np.zeros((size,a,b,c,d),dtype=np.float64)

            if LogInterp == True :

                P_comp = np.log10(P_comp)

            for i in range(a) :

                for j in range(b) :

                    for k in range(c) :

                        for l in range(d) :

                            if LogInterp == True :

                                wh, = np.where(P_comp >= np.log10(P[i,j,k,l]))

                            else :

                                wh, = np.where(P_comp >= P[i,j,k,l])

                            if wh.size != 0 :

                                # P < P_max

                                i_Pu = wh[0]
                                i_Pd = i_Pu - 1

                                if LogInterp == True :

                                    P_ref = np.log10(P[i,j,k,l])

                                else :

                                    P_ref = P[i,j,k,l]

                                whT, = np.where(T_comp >= T[i,j,k,l])

                                if whT.size != 0  :

                                    # T < T_max

                                    i_Tu = whT[0]
                                    i_Td = i_Tu - 1

                                    T_ref = T[i,j,k,l]

                                    if i_Pu == 0 and i_Tu == 0 :

                                        # Si P <= P_min et T <= T_min alors on considere T = T_min et P = P_min

                                        compo[:,i,j,k,l] = x_species[:,0,0]

                                    else :

                                        if i_Pu == 0 :

                                            # Si P <= P_min alors on considere P = P_min, interpolation sur T

                                            coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                            coeff2 = 1. - coeff1

                                            compo[2:size,i,j,k,l] = coeff1*x_species[2:size,0,i_Tu] + coeff2*x_species[2:size,0,i_Td]

                                        else :

                                            if i_Tu == 0 :

                                                # Si T <= T_min alors on considere T = T_min, interpolation sur P

                                                coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                                coeff4 = 1. - coeff3

                                                compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,0] + coeff4*x_species[2:size,i_Pd,0]

                                            else :

                                                # Si P entre P_min et P_max et T entre T_min et T_max, double interpolation

                                                coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                coeff2 = 1. - coeff1
                                                coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                                coeff4 = 1. - coeff3
                                                c14,c13,c24,c23 = coeff1*coeff4,coeff1*coeff3,coeff2*coeff4,coeff2*coeff3

                                                compo1 = c14*x_species[2:size,i_Pd,i_Tu] + c24*x_species[2:size,i_Pd,i_Td]
                                                compo2 = c13*x_species[2:size,i_Pu,i_Tu] + c23*x_species[2:size,i_Pu,i_Td]
                                                compo[2:size,i,j,k,l] = compo2 + compo1

                                        compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                        compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                else :

                                    # Si T >= T_max alors on considere T = T_max

                                    i_T = T_comp.size - 1

                                    if i_Pu == 0 :

                                        # Si P <= P_min alors on considere P = P_min, pas d'interpolation

                                        compo[2:size,i,j,k,l] = x_species[2:size,0,i_T]

                                    else :

                                        # Si P entre P_min et P_max, interpolation sur P

                                        coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                        coeff4 = 1. - coeff3

                                        compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,i_T] + coeff4*x_species[2:size,i_Pd,i_T]

                            else :

                                # Si P > P_max alors on considere P = P_max

                                i_P = P_comp.size - 1

                                whT, = np.where(T_comp >= T[i,j,k,l])

                                if whT.size != 0  :

                                    # Si T <= T_max

                                    i_Tu = whT[0]
                                    i_Td = i_Tu - 1

                                    T_ref = T[i,j,k,l]

                                    if i_Tu == 0 :

                                        # Si T < T_min alors on considere T = T_min

                                        compo[:,i,j,k,l] = x_species[:,i_P,0]

                                    else :

                                        # Si T compris entre T_min et T_max alors interpolation sur T

                                        coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                        coeff2 = 1. - coeff1

                                        compo[2:size,i,j,k,l] = coeff1*x_species[2:size,i_P,i_Tu] + coeff2*x_species[2:size,i_P,i_Td]
                                        compo[0,i,j,k,l] = (1. - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                        compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                else :

                                    # Si T > T_max alors on considere T = T_max

                                    i_T = T_comp.size - 1

                                    compo[:,i,j,k,l] = x_species[:,i_P,i_T]

                            # Si nous preferons travailler a poids moleculaire constant, Exit = False empeche le calcul
                            # du poids moleculaire sur chacune des cellules

                            M[i,j,k,l] = np.nansum(compo[:,i,j,k,l]*M_species)

    else :

        if Composition == True :

            size = species.size

            compo = np.zeros((size,a,b,c,d))

            for i in range(a) :

                for j in range(b) :

                    for k in range(c) :

                        for l in range(d) :

                            wh, = np.where(P_comp >= P[i,j,k,l])

                            if wh.size != 0 :

                                # Si P <= P_max

                                i_Pu = wh[0]
                                i_Pd = i_Pu - 1

                                P_ref = P[i,j,k,l]

                                whT, = np.where(T_comp >= T[i,j,k,l])

                                if whT.size != 0  :

                                    # Si T <= T_max

                                    i_Tu = whT[0]
                                    i_Td = i_Tu - 1

                                    T_ref = T[i,j,k,l]

                                    whQ, = np.where(Q_comp >= h2o[i,j,k,l])

                                    if whQ.size != 0 :

                                        # Si Q <= Q_max

                                        i_Qu = whQ[0]
                                        i_Qd = i_Qu - 1

                                        Q_ref = h2o[i,j,k,l]

                                        if i_Pu == 0 and i_Tu == 0 and i_Qu == 0 :

                                            # Si T <= T_min, P <= P_min et Q <= Q_min alors on considere T = T_min,
                                            # P = P_min et Q = Q_min

                                            compo[:,i,j,k,l] = x_species[:,0,0,0]

                                        else :

                                            if i_Pu == 0 :

                                                # Si P <= P_min

                                                if i_Tu != 0 and i_Qu != 0 :

                                                    # Si T et Q sont compris entre T_min et T_max ou Q_min et Q_max alors
                                                    # double interpolaton sur T et Q

                                                    coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                    coeff2 = 1. - coeff1
                                                    coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                    coeff6 = 1. - coeff5
                                                    c16,c15,c26,c25 = coeff1*coeff6,coeff1*coeff5,coeff2*coeff6,coeff2*coeff5

                                                    compo1 = c15*x_species[2:size,0,i_Tu,i_Qu] + c25*x_species[2:size,0,i_Td,i_Qu]
                                                    compo2 = c16*x_species[2:size,0,i_Tu,i_Qd] + c26*x_species[2:size,0,i_Td,i_Qd]
                                                    compo[2:size,i,j,k,l] = compo1 + compo2

                                                else :

                                                    if i_Qu == 0 :

                                                        # Si Q <= Q_min alors on considere Q = Q_min et on effectue une
                                                        # interpolation sur T

                                                        coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                        coeff2 = 1. - coeff2

                                                        compo[2:size,i,j,k,l] = coeff1*x_species[2:size,0,i_Tu,0] + coeff2*x_species[2:size,0,i_Td,0]

                                                    if i_Tu == 0 :

                                                        # Si T <= T_min alors on considere T = T_min et on effectue une
                                                        # interpolation sur Q

                                                        coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                        coeff6 = 1. - coeff5

                                                        compo[2:size,i,j,k,l] = coeff5*x_species[2:size,0,0,i_Qu] + coeff6*x_species[2:size,0,0,i_Qd]

                                                compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                                compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                            else :

                                                # Si P est compris entre P_min et P_max alors interpolation sur P

                                                coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                                coeff4 = 1. - coeff3

                                                if i_Tu == 0 and i_Qu != 0 :

                                                    # Si T <= T_min alors on considere T = T_min et on effectue une
                                                    # interpolation sur Q

                                                    coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                    coeff6 = 1. - coeff5
                                                    c36,c35,c46,c45 = coeff3*coeff6,coeff3*coeff5,coeff4*coeff6,coeff4*coeff5

                                                    compo1 = c35*x_species[2:size,i_Pu,0,i_Qu] + c45*x_species[2:size,i_Pd,0,i_Qu]
                                                    compo2 = c36*x_species[2:size,i_Pu,0,i_Qd] + c46*x_species[2:size,i_Pd,0,i_Qd]
                                                    compo[2:size,i,j,k,l] = compo1 + compo2

                                                if i_Qu == 0 and i_Tu != 0 :

                                                    # Si Q <= Q_min alors on considere Q = Q_min et on effectue une
                                                    # interpolation sur T

                                                    coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                    coeff2 = 1. - coeff1
                                                    c14,c13,c24,c23 = coeff1*coeff4,coeff1*coeff3,coeff2*coeff4,coeff2*coeff3

                                                    compo1 = c13*x_species[2:size,i_Pu,i_Tu,0] + c23*x_species[2:size,i_Pu,i_Td,0]
                                                    compo2 = c14*x_species[2:size,i_Pd,i_Tu,0] + c24*x_species[2:size,i_Pd,i_Td,0]
                                                    compo[2:size,i,j,k,l] = compo1 + compo2

                                                if i_Qu == 0  and i_Tu == 0 :

                                                    # Si T <= T_min et Q <= Q_min alors considere T = T_min et Q = Q_min

                                                    compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,0,0] + coeff4*x_species[2:size,i_Pd,0,0]

                                                if i_Qu != 0 and i_Pu != 0 :

                                                    # Si T et Q sont egalement compris entre respectivement T_min et T_max et
                                                    # Q_min et Q_max, alors triple interpolation sur P, T et Q
                                                    # Cette formulation est la plus generale et celle qui est vraiment susceptible
                                                    # d'intervenir dans les calculs de composition, les encadrements sur P, T et Q
                                                    # etant choisis en fonction de la simulation

                                                    coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                    coeff2 = 1. - coeff1
                                                    coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                    coeff6 = 1. - coeff5
                                                    c315,c415,c316,c416 = coeff3*coeff1*coeff5,coeff4*coeff1*coeff5,\
                                                                          coeff3*coeff1*coeff6,coeff4*coeff1*coeff6
                                                    c325,c425,c326,c426 = coeff3*coeff2*coeff5,coeff4*coeff2*coeff5,\
                                                                          coeff3*coeff2*coeff6,coeff4*coeff2*coeff6

                                                    compo1 = c315*x_species[2:size,i_Pu,i_Tu,i_Qu] + c415*x_species[2:size,i_Pd,i_Tu,i_Qu]
                                                    compo2 = c316*x_species[2:size,i_Pu,i_Tu,i_Qd] + c416*x_species[2:size,i_Pd,i_Tu,i_Qd]
                                                    compo3 = c325*x_species[2:size,i_Pu,i_Td,i_Qu] + c425*x_species[2:size,i_Pd,i_Td,i_Qu]
                                                    compo4 = c326*x_species[2:size,i_Pu,i_Td,i_Qd] + c426*x_species[2:size,i_Pd,i_Td,i_Qd]
                                                    compo[2:size,i,j,k,l] = compo1 + compo2 + compo3 + compo4

                                                compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                                compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                    else :

                                        # Si Q > Q_max alors on considere Q = Q_max

                                        i_Q = Q_comp.size - 1

                                        if i_Pu == 0 :

                                            # Si P <= P_min alors on considere P = P_min

                                            if i_Tu == 0 :

                                                # Si T <= T_min alors on considere T = T_min

                                                compo[:,i,j,k,l] = x_species[:,0,0,i_Q]

                                            else :

                                                # Si T est compris entre T_min et T_max alors interpolation sur T

                                                coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                coeff2 = 1. - coeff1

                                                compo[2:size,i,j,k,l] = coeff1*x_species[2:size,0,i_Tu,i_Q] + coeff2*x_species[2:size,0,i_Td,i_Q]

                                        else :

                                            # Si P est compris entre P_min et P_max alors interpolation sur P

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = 1. - coeff3

                                            if i_Tu == 0 :

                                                # Si T <= T_min alors on considere T = T_min

                                                compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,0,i_Q] + coeff4*x_species[2:size,i_Pd,0,i_Q]

                                            else :

                                                # Si T est compris entre T_min et T_max alors double interpolation sur
                                                # P et T

                                                coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                                coeff2 = 1. - coeff1
                                                c14,c13,c24,c23 = coeff1*coeff4,coeff1*coeff3,coeff2*coeff4,coeff2*coeff3

                                                compo1 = c13*x_species[2:size,i_Pu,i_Tu,i_Q] + c23*x_species[2:size,i_Pu,i_Td,i_Q]
                                                compo2 = c14*x_species[2:size,i_Pd,i_Tu,i_Q] + c24*x_species[2:size,i_Pd,i_Td,i_Q]

                                                compo[2:size,i,j,k,l] = compo1 + compo2

                                        compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                        compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                else :

                                    # Si T > T_max alors on considere T = T_max

                                    i_T = T_comp.size - 1

                                    whQ, = np.where(Q_comp >= h2o[i,j,k,l])

                                    if whQ.size != 0 :

                                        # Si Q <= Q_max

                                        i_Qu = whQ[0]
                                        i_Qd = i_Qu - 1

                                        Q_ref = h2o[i,j,k,l]

                                        if i_Pu == 0 :

                                            # Si P <= P_min alors on considere P = P_min

                                            if i_Qu == 0 :

                                                # Si Q <= Q_min alors on considere Q = Q_min

                                                compo[:,i,j,k,l] = x_species[:,0,i_T,0]

                                            else :

                                                # Si Q est compris entre Q_min et Q_max alors interpolation sur Q

                                                coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                coeff6 = 1. - coeff6

                                                compo[2:size,i,j,k,l] = coeff5*x_species[2:size,0,i_T,i_Qu] + coeff6*x_species[2:size,0,i_T,i_Qd]

                                        else :

                                            # Si P est compris entre P_min et P_max alors interpolation sur P

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = 1. - coeff3

                                            if i_Qu == 0 :

                                                # Si Q <= Q_min alors on considere Q = Q_min

                                                compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,i_T,0] + coeff4*x_species[2:size,i_Pd,i_T,0]

                                            else :

                                                # Si Q est compris entre Q_min et Q_max alors double interpolation sur P et Q

                                                coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                coeff6 = 1. - coeff5
                                                c36,c35,c46,c45 = coeff3*coeff6,coeff3*coeff5,coeff4*coeff6,coeff4*coeff5

                                                compo1 = c35*x_species[2:size,i_Pu,i_T,i_Qu] + c36*x_species[2:size,i_Pu,i_T,i_Qd]
                                                compo2 = c45*x_species[2:size,i_Pd,i_T,i_Qu] + c46*x_species[2:size,i_Pd,i_T,i_Qd]

                                                compo[2:size,i,j,k,l] = compo1 + compo2

                                        compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                        compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                    else :

                                        # Si Q > Q_max alors on considere Q = Q_max

                                        if Q_comp.size != 0 :

                                            i_Q = Q_comp.size - 1

                                        else :

                                            i_Q = 0

                                        if i_Pu == 0 :

                                            # Si P <= P_min alors on considere P = P_min

                                            compo[:,i,j,k,l] = x_species[:,0,i_T,i_Q]

                                        else :

                                            # Si P est compris entre P_min et P_max alors interpolation sur P

                                            P_ref = P[i,j,k,l]

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = 1. - coeff3

                                            compo[2:size,i,j,k,l] = coeff3*x_species[2:size,i_Pu,i_T,i_Q] + coeff4*x_species[2:size,i_Pd,i_T,i_Q]
                                            compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                            compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                            else :

                                # Si P > P_max alors on considere P = P_max

                                i_P = P_comp.size - 1

                                whT, = np.where(T_comp >= T[i,j,k,l])

                                if whT.size != 0  :

                                    # Si T <= T_max

                                    i_Tu = whT[0]
                                    i_Td = i_Tu - 1

                                    T_ref = T[i,j,k,l]

                                    whQ, = np.where(Q_comp >= h2o[i,j,k,l])

                                    if whQ.size != 0 :

                                        # Si Q <= Q_max

                                        i_Qu = whQ[0]
                                        i_Qd = i_Qu - 1

                                        Q_ref = h2o[i,j,k,l]

                                        if i_Tu == 0 :

                                            # Si T <= T_min alors on considere T = T_min

                                            if i_Qu == 0 :

                                                # Si Q <= Q_min alors on considere Q = Q_min

                                                compo[:,i,j,k,l] = x_species[:,i_P,0,0]

                                            else :

                                                # Si Q est compris entre Q_min et Q_max alors interpolation sur Q

                                                coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                coeff6 = 1. - coeff5

                                                compo[2:size,i,j,k,l] = coeff5*x_species[2:size,i_P,0,i_Qu] + coeff6*x_species[2:size,i_P,0,i_Qd]

                                        else :

                                            # Si T est compris entre T_min et T_max alors interpolation sur T

                                            coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                            coeff2 = 1. - coeff1

                                            if i_Qu == 0 :

                                                # Si Q <= Q_min alors on considere Q = Q_min

                                                compo[2:size,i,j,k,l] = coeff1*x_species[2:size,i_P,i_Tu,0] + coeff2*x_species[2:size,i_P,i_Td,0]

                                            else :

                                                # Si Q est compris entre Q_min et Q_max alors double interpolation sur T et Q

                                                coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                                coeff6 = 1. - coeff5
                                                c16,c15,c26,c25 = coeff1*coeff6,coeff1*coeff5,coeff2*coeff6,coeff2*coeff5

                                                compo1 = c15*x_species[2:size,i_P,i_Tu,i_Qu] + c25*x_species[2:size,i_P,i_Td,i_Qu]
                                                compo2 = c16*x_species[2:size,i_P,i_Tu,i_Qd] + c26*x_species[2:size,i_P,i_Td,i_Qd]

                                                compo[2:size,i,j,k,l] = compo1 + compo2

                                        compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                        compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                    else :

                                        # Si Q > Q_max alors on considere Q = Q_max

                                        if Q_comp.size != 0 :

                                            i_Q = Q_comp.size - 1

                                        else :

                                            i_Q = 0

                                        if i_Tu == 0 :

                                            # Si T <= T_min alors on considere T = T_min

                                            compo[:,i,j,k,l] = x_species[:,i_P,0,i_Q]

                                        else :

                                            # Si T est compris entre T_min et T_max alors interpolation sur T

                                            T_ref = T[i,j,k,l]

                                            coeff1 = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])
                                            coeff2 = 1. - coeff1

                                            compo[2:size,i,j,k,l] = coeff1*x_species[2:size,i_P,i_Tu,i_Q] + coeff2*x_species[2:size,i_P,i_Td,i_Q]
                                            compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                            compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                else :

                                    # Si T > T_max alors on considere T = T_max

                                    i_T = T_comp.size - 1

                                    whQ, = np.where(Q_comp >= h2o[i,j,k,l])

                                    if whQ.size != 0 :

                                        # Si Q <= Q_max

                                        i_Qu = whQ[0]
                                        i_Qd = i_Qu - 1

                                        Q_ref = h2o[i,j,k,l]

                                        if i_Qu == 0 :

                                            # Si Q <= Q_min alors on considere Q = Q_min

                                            compo[:,i,j,k,l] = x_species[:,i_P,i_T,0]

                                        else :

                                            # Si Q est compris entre Q_min et Q_max alors interpolation sur Q

                                            coeff5 = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])
                                            coeff6 = 1. - coeff5

                                            compo[2:size,i,j,k,l] = coeff5*x_species[2:size,i_P,i_T,i_Qu] + coeff6*x_species[2:size,i_P,i_T,i_Qd]
                                            compo[0,i,j,k,l] = (1 - np.nansum(compo[2:size,i,j,k,l]))/(ratio + 1.)
                                            compo[1,i,j,k,l] = compo[0,i,j,k,l]*ratio

                                    else :

                                        # Si Q > Q_max alors on considere Q = Q_max

                                        if Q_comp.size != 0 :

                                            i_Q = Q_comp.size - 1

                                        else :

                                            i_Q = 0

                                        compo[:,i,j,k,l] = x_species[:,i_P,i_T,i_Q]

                            M[i,j,k,l] = np.nansum(compo[:,i,j,k,l]*M_species)

    if Composition == False :

        M += M_atm

    # Une fois la composition dans chaque cellule des donnees GCM calculee, nous avons l'information manquante sur le
    # poids moleculaire moyen et donc sur la hauteur d'echelle locale. Nous pouvons alors transformer l'echelle de
    # pression en echelle d'altitude

    for pres in range(b) :

        if pres == 0 :

            z[:,0,:,:] = 0.
            Mass = np.zeros((a,c,d))
            g = g0

        else :

            # Premiere estmiation de l'altitude avec l'acceleration de la pesanteur de la couche precedente

            g = g0 + Mass*G/(Rp + z[:,pres-1,:,:])**2
            a_z = -(1+z[:,pres-1,:,:]/Rp)*R_gp*(T[:,pres,:,:]-T[:,pres-1,:,:])/((M[:,pres,:,:]+M[:,pres-1,:,:])/2.*g*\
                np.log(T[:,pres,:,:]/T[:,pres-1,:,:]))*np.log(P[:,pres,:,:]/P[:,pres-1,:,:])
            dz = a_z*(1+z[:,pres-1,:,:]/Rp)/(1-a_z/Rp)

            wht,whla,whlo = np.where(T[:,pres,:,:]==T[:,pres-1,:,:])

            if wht.size != 0 :

                for tw in range(wht.size) :

                    dz[wht[tw],whla[tw],whlo[tw]] = \
                        -R_gp*T[wht[tw],pres,whla[tw],whlo[tw]]/(M[wht[tw],pres,whla[tw],whlo[tw]]*g[wht[tw],whla[tw],whlo[tw]])*\
                                    np.log(P[wht[tw],pres,whla[tw],whlo[tw]]/P[wht[tw],pres-1,whla[tw],whlo[tw]])

            z[:,pres,:,:] = z[:,pres-1,:,:] + dz

            # On incremente petit a petit la masse atmospherique

            Mass += P[:,pres,:,:]/(R_gp*T[:,pres,:,:])*M[:,pres,:,:]*4/3.*np.pi*((Rp + z[:,pres,:,:])**3 - (Rp + z[:,pres-1,:,:])**3)

    return compo, M, z


########################################################################################################################
########################################################################################################################


def Boxes_conversion(P,T,h2o,gen,z,compo,delta_z,Rp,h,hmax,dim,g0,M_atm,number,T_comp,P_comp,Q_comp,x_species,M_species,ratio,rank,Upper,\
        Marker=False,Clouds=False,Composition=False,Middle=False) :

    a, b, c, d = np.shape(P)
   
    data_convert = np.zeros((number,a,dim,c,d))
    
    if Clouds == True : 
        sh_c = np.shape(gen)
	c_number = sh_c[0]

    if Composition == False :
        data_convert[number - 1,:,:,:,:] += M_atm
    else : 
        sh_comp = np.shape(compo)
	size = sh_comp[0]

    Mass = np.zeros((a,c,d))
    Reformate = False

    first = np.zeros((a,c,d))
    coeff1 = np.zeros((a,c,d))
    coeff5 = np.zeros((a,c,d))
    
    if rank == 0 : 
        bar = ProgressBar(dim,'State of the conversion :')

    for i_z in range(dim) :

        # Si la fonction Middle est selectionnee, le code va formater la grille cylindrique de maniere a ce que le
        # premier point corresponde aux donnees de surface tandis que les autres points correspondront aux donnees
        # des milieux de couche.

        if Middle == False :

            z_ref = i_z*delta_z

        else :

            if i_z == 0 :

                z_ref = 0.

            else :

                z_ref = (i_z-0.5)*delta_z

        if z_ref >= hmax :

            Reformate = True

        for t in range(a) :

            for lat in range(c) :

                for long in range(d) :

                    # Nous cherchons l'intervalle dans lequel se situe le point d'altitude considere

                    wh, = np.where(z[t,:,lat,long] >= z_ref)

                    # Si le point en question n'est pas au dessus du toit du modele a cette lattitude et a cette longitude

                    if wh.size != 0 :

                        # Si z_ref n'est pas un point d'altitude de la maille de depart

                        if z[t,wh[0],lat,long] != z_ref :

                            up = wh[0]
                            dn = up - 1

                            coeffa = (z[t,up,lat,long] - z_ref)/(z[t,up,lat,long] - z[t,dn,lat,long])
                            coeffb = (z_ref - z[t,dn,lat,long])/(z[t,up,lat,long] - z[t,dn,lat,long])

                            # Interpolation pour la pression puis la temperature

                            data_convert[0,t,i_z,lat,long] = np.exp(coeffa*np.log(P[t,dn,lat,long]) + coeffb*np.log(P[t,up,lat,long]))
                            data_convert[1,t,i_z,lat,long] = coeffa*T[t,dn,lat,long] + coeffb*T[t,up,lat,long]

                            if Marker == True :

                                # Interpolation sur la fraction massique du marqueur

                                data_convert[2,t,i_z,lat,long] = coeffa*h2o[t,dn,lat,long] + coeffb*h2o[t,up,lat,long]

                                if Clouds == True :

                                    # Pour chaque aerosol, on effectue l'interpolation

                                    for c_num in range(c_number) :

                                        data_convert[3+c_num,t,i_z,lat,long] = coeffa*gen[c_num,t,dn,lat,long] + coeffb*gen[c_num,t,up,lat,long]

                                    if Composition == True :

                                        # Interpolation sur la composition chimique et donc les fractions molaires, la somme de ces fractions
                                        # doit etre egale a 1

                                        for i in range(size) :

                                            data_convert[3+c_number+i,t,i_z,lat,long] = coeffa*compo[i,t,dn,lat,long] +coeffb*compo[i,t,up,lat,long]

                                        # Calcul de la masse molaire moyenne a partir de la composition chimique

                                        data_convert[3+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[3+c_number:3+c_number+size,t,i_z,lat,long]*M_species)

                                else :

                                    if Composition == True :

                                        # Interpolation sur la composition chimique et donc les fractions molaires, la somme de ces fractions
                                        # doit etre egale a 1

                                        for i in range(size) :

                                            data_convert[3+i,t,i_z,lat,long] = coeffa*compo[i,t,dn,lat,long] +coeffb*compo[i,t,up,lat,long]

                                        # Calcul de la masse molaire moyenne a partir de la composition chimique

                                        data_convert[3+size,t,i_z,lat,long] = np.nansum(data_convert[3:3+size,t,i_z,lat,long]*M_species)

                            else :

                                if Clouds == True :

                                    # Pour chaque aerosol, on effectue l'interpolation

                                    for c_num in range(c_number) :

                                        data_convert[2+c_num,t,i_z,lat,long] = coeffa*gen[c_num,t,dn,lat,long] + coeffb*gen[c_num,t,up,lat,long]

                                    if Composition == True :

                                        for i in range(size) :

                                            data_convert[2+c_number+i,t,i_z,lat,long] = coeffa*compo[i,t,dn,lat,long] +coeffb*compo[i,t,up,lat,long]

                                        data_convert[2+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[2+c_number:2+c_number+size,t,i_z,lat,long]*M_species)

                                else :

                                    if Composition == True :

                                        # Interpolation sur la composition chimique et donc les fractions molaires, la somme de ces fractions
                                        # doit etre egale a 1

                                        for i in range(size) :

                                            data_convert[2+i,t,i_z,lat,long] = coeffa*compo[i,t,dn,lat,long] +coeffb*compo[i,t,up,lat,long]

                                        # Calcul de la masse molaire moyenne a partir de la composition chimique

                                        data_convert[2+size,t,i_z,lat,long] = np.nansum(data_convert[2:2+size,t,i_z,lat,long]*M_species)

                            # Si le point considere n'est pas le premier, et donc, le point de surface, on calcule la masse d'atmosphere
                            # a pendre en compte ensuite dans l'extrapolation

                            if i_z != 0 :

                                Mass[t,lat,long] += data_convert[0,t,i_z,lat,long]/(R_gp*data_convert[1,t,i_z,lat,long])*\
                                        data_convert[number-1,t,i_z,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                        # Sinon pas d'interpolation necessaire

                        else :

                            data_convert[0,t,i_z,lat,long] = P[t,wh[0],lat,long]
                            data_convert[1,t,i_z,lat,long] = T[t,wh[0],lat,long]

                            if Marker == True :

                                data_convert[2,t,i_z,lat,long] = h2o[t,wh[0],lat,long]

                                if Clouds == True :

                                    for c_num in range(c_number) :

                                        data_convert[3+c_num,t,i_z,lat,long] = gen[c_num,t,wh[0],lat,long]

                                    if Composition == True :

                                        for i in range(size) :

                                            data_convert[3+c_number+i,t,i_z,lat,long] = compo[i,t,wh[0],lat,long]

                                        data_convert[3+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[3+c_number:3+c_number+size,t,i_z,lat,long]*M_species)

                                else :

                                    if Composition == True :

                                        for i in range(size) :

                                            data_convert[3+i,t,i_z,lat,long] = compo[i,t,wh[0],lat,long]

                                        data_convert[3+size,t,i_z,lat,long] = np.nansum(data_convert[3:3+size,t,i_z,lat,long]*M_species)

                            else :

                                if Clouds == True :

                                    for c_num in range(c_number) :

                                        data_convert[2+c_num,t,i_z,lat,long] = gen[c_num,t,wh[0],lat,long]

                                    if Composition == True :

                                        for i in range(size) :

                                            data_convert[2+c_number+i,t,i_z,lat,long] = compo[i,t,wh[0],lat,long]

                                        data_convert[2+c_number+size,t,i_z,lat,long] = np.nansum(data_convert[2+c_number:2+c_number+size,t,i_z,lat,long]*M_species)

                                else :

                                    if Composition == True :

                                        for i in range(size) :

                                            data_convert[2+i,t,i_z,lat,long] = compo[i,t,wh[0],lat,long]

                                        data_convert[2+size,t,i_z,lat,long] = np.nansum(data_convert[2:2+size,t,i_z,lat,long]*M_species)

                            if i_z != 0 :

                                Mass[t,lat,long] += data_convert[0,t,i_z,lat,long]/(R_gp*data_convert[1,t,i_z,lat,long])*\
                                    data_convert[number-1,t,i_z,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                    # Si le point d'altitude est plus eleve que le toit du modele a cette lattitude et cette longitude
                    # il nous faut extrapoler

                    else :

                        # Nous avons besoin d'une temperature de reference pour trouver la composition sur le dernier point
                        # en altitude, suivant le type d'extrapolation, nous ne pouvons pas l'identifier a celle deja calculee
                        # et nous preferons l'executer a partir des donnees d'equilibre que sur des resultats d'interpolation

                        if Reformate == False :

                            data_convert[1,t,i_z,lat,long] = T[t,b-1,lat,long]

                        else :

                            # En fonction du type d'exrapolation, nous pouvons ou non optimiser l'interpolation sur la
                            # composition chimique en haute atmosphere

                            if Upper == "Isotherme" :

                                data_convert[1,t,i_z,lat,long] = T[t,b-1,lat,long]

                            if Upper ==  "Isotherme_moyen" :

                                data_convert[1,t,i_z,lat,long] = T_mean

                            if Upper == "Maximum_isotherme" :

                                data_convert[1,t,i_z,lat,long] = T_max

                            if Upper == "Minimum_isotherme" :

                                data_convert[1,t,i_z,lat,long] = T_min

                        # On estime la pression au dela du toit a partir de la temperature choisie

                        g = g0 + Mass[t,lat,long]*G/(Rp + i_z*delta_z)**2

                        data_convert[0,t,i_z,lat,long] = data_convert[0,t,i_z-1,lat,long]*np.exp(-data_convert[number-1,t,i_z-1,lat,long]*g*\
                                delta_z/(R_gp*data_convert[1,t,i_z,lat,long])*1/((1+z_ref/Rp)*(1+(z_ref-delta_z)/Rp)))

                        T_ref = data_convert[1,t,i_z,lat,long]

                        # On calcule les coefficients d'interpolation

                        if first[t,lat,long] == 0 :

                            whT, = np.where(T_comp >= T[t,b-1,lat,long])

                            if whT.size != 0  :

                                # Si T_ref < T_max

                                if whT[0] != 0 :

                                    # Si T_ref compris entre T_min et T_max alors interpolation sur T

                                    i_Tu = whT[0]
                                    i_Td = i_Tu - 1

                                    coeff1[t,lat,long] = (T_ref - T_comp[i_Td])/(T_comp[i_Tu] - T_comp[i_Td])

                                else :

                                    # Si T_ref <= T_min, alors en identifiant les indices et avec des coefficient de 0.5
                                    # le resultat de l'interpolation donne les valeurs pour T = T_min

                                    i_Tu = 0
                                    i_Td = 0

                                    coeff1[t,lat,long] = 0.5

                            else :

                                # T > T_max, alors en identifiant les indices et avec des coefficients de 0.5 le resultat
                                # de l'interpolation donne les valeurs pour T_max

                                i_Tu = T_comp.size - 1
                                i_Td = T_comp.size - 1

                                coeff1[t,lat,long] = 0.5

                        # On incremente toujours la masse atmospherique pour la latitude et la longitude donnee, les
                        # ce point est a modifier

                        Mass[t,lat,long] += data_convert[0,t,i_z-1,lat,long]/(R_gp*data_convert[1,t,i_z-1,lat,long])*\
                                data_convert[number-1,t,i_z-1,lat,long]*4/3.*np.pi*((Rp + i_z*delta_z)**3 - (Rp + (i_z - 1)*delta_z)**3)

                        P_ref = data_convert[0,t,i_z,lat,long]

                        if Marker == True :

                            data_convert[2,t,i_z,lat,long] = h2o[t,b-1,lat,long]

                            if Clouds == True :

                                data_convert[3:3+c_number,t,i_z,lat,long] = gen[:,t,b-1,lat,long]

                                # Hormis la composition, on identifie les couches superieures au toit de l'atmosphere
                                # pour les fractions massiques en marqueur et aerosols

                                if Composition == True :

                                    whP, = np.where(P_comp >= data_convert[0,t,i_z,lat,long])

                                    if first[t,lat,long] == 0 :

                                        whQ, = np.where(Q_comp >= data_convert[2,t,i_z,lat,long])

                                    if whP.size != 0 :

                                        # Si P_ref < P_max

                                        if whP[0] != 0 :

                                            # Si P_ref compris entre P_min et P_max, alors double interpolation sur T et P

                                            i_Pu = whP[0]
                                            i_Pd = i_Pu - 1

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = (P_comp[i_Pu] - P_ref)/(P_comp[i_Pu] - P_comp[i_Pd])

                                            if first[t,lat,long] == 0 :

                                                if whQ.size != 0 :

                                                    # Si Q_ref < Q_max

                                                    if whQ[0] == 0 :

                                                        # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                        # resultat est celui de Q_ref = Q_min

                                                        i_Qu = 0
                                                        i_Qd = 0

                                                        coeff5[t,lat,long] = 0.5

                                                    else :

                                                        # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                        # sur P, T et Q

                                                        i_Qu = whQ[0]
                                                        i_Qd = i_Qu - 1

                                                        coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                                else :

                                                    # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                    # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                    i_Qu = Q_comp.size - 1
                                                    i_Qd = Q_comp.size - 1

                                                    coeff5[t,lat,long] = 0.5


                                            compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                            compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td,i_Qd]
                                            compo3 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]
                                            compo4 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td,i_Qu]

                                            compo5 = coeff3*compo2 + coeff4*compo1
                                            compo6 = coeff3*compo4 + coeff4*compo3

                                            compos = coeff5[t,lat,long]*compo6 + (1-coeff5[t,lat,long])*compo5

                                        else :

                                            # Si P_ref <= P_min

                                            i_Pd = 0

                                            if first[t,lat,long] == 0 :

                                                if whQ.size != 0 :

                                                    # Si Q_ref < Q_max

                                                    if whQ[0] == 0 :

                                                        # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                        # resultat est celui de Q_ref = Q_min

                                                        i_Qu = 0
                                                        i_Qd = 0

                                                        coeff5[t,lat,long] = 0.5

                                                    else :

                                                        # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                        # sur P, T et Q

                                                        i_Qu = whQ[0]
                                                        i_Qd = i_Qu - 1

                                                        coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                                else :

                                                    # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                    # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                    i_Qu = Q_comp.size - 1
                                                    i_Qd = Q_comp.size - 1

                                                    coeff5[t,lat,long] = 0.5

                                            compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                            compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]

                                            compos = coeff5[t,lat,long]*compo2 + (1-coeff5[t,lat,long])*compo1

                                    else :

                                        # Si P_ref > P_max

                                        i_Pd = P_comp.size - 1

                                        if first[t,lat,long] == 0 :

                                            if whQ.size != 0 :

                                                # Si Q_ref < Q_max

                                                if whQ[0] == 0 :

                                                    # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                    # resultat est celui de Q_ref = Q_min

                                                    i_Qu = 0
                                                    i_Qd = 0

                                                    coeff5[t,lat,long] = 0.5

                                                else :

                                                    # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                    # sur P, T et Q

                                                    i_Qu = whQ[0]
                                                    i_Qd = i_Qu - 1

                                                    coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                            else :

                                                # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                i_Qu = Q_comp.size - 1
                                                i_Qd = Q_comp.size - 1

                                                coeff5[t,lat,long] = 0.5

                                        compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                        compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]

                                        compos = coeff5[t,lat,long]*compo2 + (1-coeff5[t,lat,long])*compo1

                                    # On s'assure que la somme des fractions molaires est bien egale a 1

                                    compoH2 = (1 - np.nansum(compos))/(ratio + 1.)
                                    compoHe = compoH2*ratio

                                    data_convert[3+c_number,t,i_z,lat,long] = compoH2
                                    data_convert[4+c_number,t,i_z,lat,long] = compoHe
                                    data_convert[5+c_number:number-1,t,i_z,lat,long] = compos
                                    data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[3+c_number:number-1,t,i_z,lat,long]*\
                                        M_species)

                            else :

                                if Composition == True :

                                    whP, = np.where(P_comp >= data_convert[0,t,i_z,lat,long])
                                    whQ, = np.where(Q_comp >= data_convert[2,t,i_z,lat,long])

                                    if whP.size != 0 :

                                        # Si P_ref < P_max

                                        if first[t,lat,long] == 0 :

                                            if whP[0] != 0 :

                                                # Si P_ref compris entre P_min et P_max, alors double interpolation sur T et P

                                                i_Pu = whP[0]
                                                i_Pd = i_Pu - 1

                                                coeff3[t,lat,long] = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])

                                                if whQ.size != 0 :

                                                    # Si Q_ref < Q_max

                                                    if whQ[0] == 0 :

                                                        # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                        # resultat est celui de Q_ref = Q_min

                                                        i_Qu = 0
                                                        i_Qd = 0

                                                        coeff5[t,lat,long] = 0.5

                                                    else :

                                                        # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                        # sur P, T et Q

                                                        i_Qu = whQ[0]
                                                        i_Qd = i_Qu - 1

                                                        coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                                else :

                                                    # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                    # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                    i_Qu = Q_comp.size - 1
                                                    i_Qd = Q_comp.size - 1

                                                    coeff5[t,lat,long] = 0.5


                                            compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                            compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td,i_Qd]
                                            compo3 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]
                                            compo4 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td,i_Qu]

                                            compo5 = coeff3*compo2 + coeff4*compo1
                                            compo6 = coeff3*compo4 + coeff4*compo3

                                            compos = coeff5[t,lat,long]*compo6 + (1-coeff5[t,lat,long])*compo5

                                        else :

                                            # Si P_ref <= P_min

                                            i_Pd = 0

                                            if first[t,lat,long] == 0 :

                                                if whQ.size != 0 :

                                                    # Si Q_ref < Q_max

                                                    if whQ[0] == 0 :

                                                        # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                        # resultat est celui de Q_ref = Q_min

                                                        i_Qu = 0
                                                        i_Qd = 0

                                                        coeff5[t,lat,long] = 0.5

                                                    else :

                                                        # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                        # sur P, T et Q

                                                        i_Qu = whQ[0]
                                                        i_Qd = i_Qu - 1

                                                        coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                                else :

                                                    # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                    # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                    i_Qu = Q_comp.size - 1
                                                    i_Qd = Q_comp.size - 1

                                                    coeff5[t,lat,long] = 0.5

                                            compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                            compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]

                                            compos = coeff5[t,lat,long]*compo2 + (1-coeff5[t,lat,long])*compo1

                                    else :

                                        # Si P_ref > P_max

                                        i_Pd = P_comp.size - 1

                                        if first[t,lat,long] == 0 :

                                            if whQ.size != 0 :

                                                # Si Q_ref < Q_max

                                                if whQ[0] == 0 :

                                                    # Si Q <= Q_min, on identifie les indices et les coefficients a 0.5, le
                                                    # resultat est celui de Q_ref = Q_min

                                                    i_Qu = 0
                                                    i_Qd = 0

                                                    coeff5[t,lat,long] = 0.5

                                                else :

                                                    # Si Q compris entre Q_min et Q_max, alors triple interpolation possible
                                                    # sur P, T et Q

                                                    i_Qu = whQ[0]
                                                    i_Qd = i_Qu - 1

                                                    coeff5[t,lat,long] = (Q_ref - Q_comp[i_Qd])/(Q_comp[i_Qu] - Q_comp[i_Qd])

                                            else :

                                                # Si Q_ref > Q_max alors on identifie les indices et les coefficients
                                                # a 0.5, le resultat de l'interpolation est celui pour Q = Q_max

                                                i_Qu = Q_comp.size - 1
                                                i_Qd = Q_comp.size - 1

                                                coeff5[t,lat,long] = 0.5

                                        compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qd] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qd]
                                        compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu,i_Qu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td,i_Qu]

                                        compos = coeff5[t,lat,long]*compo2 + (1-coeff5[t,lat,long])*compo1

                                    # On s'assure que la somme des fractions molaires est bien egale a 1

                                    compoH2 = (1 - np.nansum(compos))/(ratio + 1.)
                                    compoHe = compoH2*ratio
                                    data_convert[3,t,i_z,lat,long] = compoH2
                                    data_convert[4,t,i_z,lat,long] = compoHe
                                    data_convert[5:number-1,t,i_z,lat,long] = compos
                                    data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[3:number-1,t,i_z,lat,long]*\
                                        M_species)

                        else :

                            if Clouds == True :

                                data_convert[2:2+c_number,t,i_z,lat,long] = gen[:,t,b-1,lat,long]

                                if Composition == True :

                                    wh, = np.where(P_comp >= data_convert[0,t,i_z,lat,long])

                                    if wh.size != 0 :

                                        # Si P < P_max

                                        if wh[0] != 0 :

                                            # Si P_ref est compris entre P_min et P_max, double interpolation sur T et P

                                            i_Pu = wh[0]
                                            i_Pd = i_Pu - 1

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = (P_comp[i_Pu] - P_ref)/(P_comp[i_Pu] - P_comp[i_Pd])

                                        else :

                                            # Si P_ref <= P_min

                                            i_Pu = 0
                                            i_Pd = 0

                                            coeff3 = 0.5
                                            coeff4 = 0.5

                                        compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td]
                                        compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td]

                                        compos = coeff3*compo2 + coeff4*compo1

                                    else :

                                        # Si P_ref > P_max

                                        i_Pd = P_comp.size - 1

                                        compos = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td]

                                    compoH2 = (1 - np.nansum(compos))/(ratio + 1.)
                                    compoHe = compoH2*ratio

                                    # On s'assure que la somme des fractions molaires est bien egale a 1

                                    data_convert[2+c_number,t,i_z,lat,long] = compoH2
                                    data_convert[3+c_number,t,i_z,lat,long] = compoHe
                                    data_convert[4+c_number:number-1,t,i_z,lat,long] = compos
                                    data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[2+c_number:number-1,t,i_z,lat,long]*\
                                        M_species)

                            else :

                                if Composition == True :

                                    wh, = np.where(P_comp >= data_convert[0,t,i_z,lat,long])

                                    if wh.size != 0 :

                                        # Si P < P_max

                                        if wh[0] != 0 :

                                            # Si P_ref est compris entre P_min et P_max, double interpolation sur T et P

                                            i_Pu = wh[0]
                                            i_Pd = i_Pu - 1

                                            coeff3 = (P_ref - P_comp[i_Pd])/(P_comp[i_Pu] - P_comp[i_Pd])
                                            coeff4 = (P_comp[i_Pu] - P_ref)/(P_comp[i_Pu] - P_comp[i_Pd])

                                        else :

                                            # Si P_ref <= P_min

                                            i_Pu = 0
                                            i_Pd = 0

                                            coeff3 = 0.5
                                            coeff4 = 0.5

                                        compo1 = coeff1[t,lat,long]*x_species[2:size,i_Pd,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td]
                                        compo2 = coeff1[t,lat,long]*x_species[2:size,i_Pu,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pu,i_Td]

                                        compos = coeff3*compo2 + coeff4*compo1

                                    else :

                                        # Si P_ref > P_max

                                        i_Pd = P_comp.size - 1

                                        compos = coeff1*x_species[2:size,i_Pd,i_Tu] + (1-coeff1[t,lat,long])*x_species[2:size,i_Pd,i_Td]

                                    compoH2 = (1 - np.nansum(compos))/(ratio + 1.)
                                    compoHe = compoH2*ratio

                                    # On s'assure que la somme des fractions molaires est bien egale a 1

                                    data_convert[2,t,i_z,lat,long] = compoH2
                                    data_convert[3,t,i_z,lat,long] = compoHe
                                    data_convert[4:number-1,t,i_z,lat,long] = compos
                                    data_convert[number-1,t,i_z,lat,long] = np.nansum(data_convert[2:number-1,t,i_z,lat,long]*\
                                        M_species)

                        first[t,lat,long] = 1
	if rank == 0 : 
	    bar.animate(i_z+1)

    return data_convert


########################################################################################################################
########################################################################################################################

"""
    CYLINDRIC_MATRIX_PARAMETER

    Produit la matrice cylindrique de reference a partir de laquelle nous allons construire les tableaux de temperature
    de pression, de fraction molaire, de fraction massique, de concentration molaire et de concentration massique. Cette
    matrice tient desormais compte de la rotation de l'exoplanete, de son inclinaison ou de son obliquite. Seules les 
    valeurs positives de l'obliquite ont ete testees pour l'instant, dans le cas d'une obliquite negative, il suffit d'
    inevrser la matrice sur le chemin optique. 

"""

########################################################################################################################
########################################################################################################################


def cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,x_step,z_array,phi_rot,\
                                   phi_obli,reso_long,reso_lat,rank,number_rank,obliquity=False,Middle=False) :

    # On definit un r maximal qui est la somme du rayon planetaire et du toit de l'atmosphere, on en deduit une valeur
    # entiere et qui est un multiple du pas en r

    rmax_round = int(Rp + h - h%r_step + r_step)
    rmax_range = int(h - h%r_step + r_step)

    # On calcule la distance maximale que peut parcourir un rayon lumineux rasant comme un entier et un multiple du pas
    # en x, L est alors la moitie de cette distance et L_round est le nombre de pas

    L = np.sqrt((Rp+h)**2 - (Rp+0.5*r_step)**2) - (np.sqrt((Rp+h)**2 - (Rp+0.5*r_step)**2))%x_step + x_step
    L_round = int(L/float(x_step))+1
    
    # Pour optimiser le calcul parallele, nous ne repartissons pas regulierement les couches a traiter par chaque coeur
    # les premieres etant plus longues a etre calculees
    
    n_level_rank = np.array([],dtype=np.int)
    n = rank
    extra = 5
    while n <= z_array.size - 1 : 
        n_level_rank = np.append(n_level_rank,n) 
	if n == z_array.size - 1 : 
	    for i_extra in range(extra) : 
	        n_level_rank = np.append(n_level_rank,n+i_extra+1)
	n += number_rank	 

    # p pour la latitude, q pour la longitude, z pour l'altitude
    # Le coeur qui s'occupera de la toute derniere couche aura 5 points fictifs supplementaires charges de deliminter 
    # l'echelle en altiude des tableaux. Le 5 est arbitraire
 
    p_grid = np.ones((int(n_level_rank.size) ,theta_number , 2*L_round + 21),dtype=np.int)*(-1)
    q_grid = np.ones((int(n_level_rank.size) ,theta_number , 2*L_round + 21),dtype=np.int)*(-1)
    z_grid = np.ones((int(n_level_rank.size) ,theta_number , 2*L_round + 21),dtype=np.int)*(-1)
        

    if rank == 0 : 
        if obliquity == True : 
            bar = ProgressBar(np.int(n_level_rank.size*theta_number),'Transposition on cylindric stitch (with obliquity)')
	else : 
	    bar = ProgressBar(np.int(n_level_rank.size*theta_number/2.),'Transposition on cylindric stitch')

    for n_level in range(n_level_rank.size) :

        r_level = Rp + n_level_rank[n_level]*r_step
	
	# Si les points de la maille spherique correspondent aux proprietes en milieu de couche, alors il faut tenir
        # compte de la demi-epaisseur de couche dans les calculs de correspondance

        if Middle == True :
            r = r_level + r_step/2.
        else :
            r = r_level
	
	r_range = n_level

        if obliquity == False :

            for i in range(0,theta_number/2+1) :
                theta_range = i
                theta = i*theta_step
                stop = 0

                for x_range in range(0,2*L_round + 21) :

                    # x est la distance au centre et peut donc etre negatif comme positif, le 0 etant au terminateur
                    x = (x_range - L_round - 10)*x_step
                    # rho est la distance au centre de l'exoplanete du point de maille considere
                    rho = np.sqrt(r**2 + x**2)
                    # alpha est la longitude correspondante
                    alpha = math.atan2(r*np.cos(theta),x) + phi_rot

                    # Les points de longitude allant de 0 a reso_long, le dernier point etant le meme que le premier, tandis qu'en
                    # angle ils correspondent a -pi a pi (pour le dernier), nous devons renormaliser la longitude

                    if alpha > np.pi :

                        alpha = -np.pi + alpha%(np.pi)

                    if alpha < -np.pi :

                        alpha = np.pi + alpha%(-np.pi)

                    # delta est la latitude correspondante

                    delta = np.arcsin((r*np.sin(theta))/(rho))
                    #print(r,theta,x,rho,delta,alpha)

                    if theta_range == 0 :

                        if rho <= Rp + h :

                            # Tant que le point considere est dans l'atmosphere

                            if Middle == False :

                                # Si on est au dessus ou sur la premiere inter-couche

                                if (rho - Rp) >= z_array[1] :

                                    # Si r est compris entre z_step et le toit de l'atmosphere

                                    if (rho - Rp) < z_array[z_array.size-1] :

                                        wh, = np.where(z_array > (rho - Rp))
                                        up = wh[0]
                                        down = wh[0]-1

                                        # Sans Middle, tous les points entre un milieu de couche et le suivant sont identifies
                                        # au niveau inter-couche qui les separe

                                        if (rho - Rp) - z_array[down] <= (z_array[down] + z_array[up])/2. :
                                            z = down
                                        else :
                                            z = up
                                    else :

                                        z = z_array.size - 1

                                else :

                                    # Si le point est compris entre la surface et le premier inter-couche, si r est en dessous
                                    # du milieu de la premiere couche, alors il est identifie a la surface, sinon, a la
                                    # premiere inter-couche

                                    if (rho - Rp) <= z_array[1]/2. :
                                        z = 0
                                    else :
                                        z = 1

                            else :

                                # z_array donne l'echelle d'altitude et les niveaux inter-couches, sur une maille Middle
                                # l'indice 0 correspond a la surface et les autres aux milieux de couche, en cherchant
                                # l'indice pour lesquel z_array est superieur a r, on identifie directement l'indice
                                # correspondant dans la maille Middle

                                if (rho - Rp) < h :

                                    wh, = np.where(z_array > (rho - Rp))
                                    z = wh[0]

                                else :

                                    z = z_array.size - 1

                    if rho <= Rp + h :

                        # A partir de la longitude, on en deduit l'indice q correspondant dans la maille spherique, cet
                        # indice doit evoluer entre 0 (alpha -pi) et reso_long (alpha pi), sachant que le premier et le
                        # dernier point sont identiques

                        if alpha%alpha_step < alpha_step/2. :
                            alpha_norm = alpha - alpha%alpha_step
                            q = reso_long/2 + int(round(alpha_norm*reso_long/(2*np.pi)))
                        else :
                            alpha_norm = alpha - alpha%alpha_step + alpha_step
                            q = reso_long/2 + int(round(alpha_norm*reso_long/(2*np.pi)))

                        # A partir de la latitude on en deduit l'indice p correpondant dans la maille spherique, cet indice
                        # doit evoluer entre 0 (-pi/2) et reso_lat (pi/2)

                        if delta%delta_step < delta_step/2. :
                            delta_norm = delta - delta%delta_step
                            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))
                        else :
                            delta_norm = delta - delta%delta_step + delta_step
                            p = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))

                        if q == reso_long :
                            q = 0

                        p_grid[r_range,theta_range,x_range] = p
                        if theta_range != theta_number/4 or theta_range != 3*theta_number/4 :
                            q_grid[r_range,theta_range,x_range] = q
                        else :
                            q_grid[r_range,theta_range,x_range] = q_grid[r_range,theta_range,x_range-1]

                        # Etant donne la symetrie, on generaliser z pour tous les theta et en deduire les indices en longitude
                        # et en latitude par rotation, les longitudes sont les memes pour theta et theta_number - theta

                        if theta_range == 0 :
                            z_grid[r_range,theta_range,x_range] = z
                        else :
                            z_grid[r_range,theta_range,x_range] = z_grid[r_range,0,x_range]
                            if theta_range != theta_number/4 or theta_range != 3*theta_number/4 :
                                q_grid[r_range,theta_number - theta_range,x_range] = q
                            else :
                                q_grid[r_range,theta_number - theta_range,x_range] = q_grid[r_range,theta_range,x_range-1]
                            p_grid[r_range,theta_number - theta_range,x_range] = reso_lat - p
                            z_grid[r_range,theta_number - theta_range,x_range] = z_grid[r_range,0,x_range]

                if rank == 0 : 
	            bar.animate(r_range*theta_number/2.+theta_range)
	    
        else :

            for i in range(0,theta_number) :
                theta_range = i
                theta = i*theta_step
                alpha_o_ref = -1
                begin = 0
                inv = 0
                refrac = 0

                for x_range in range(0,2*L_round + 21) :

                    # x est la distance au centre et peut donc etre negatif comme positif, le 0 etant au terminateur
                    x = (x_range - L_round - 10)*x_step
                    # rho est la distance au centre de l'exoplanete du point de maille considere
                    rho = np.sqrt(r**2 + x**2)
                    # alpha est la longitude correspondante
                    alpha = math.atan2(r*np.cos(theta),x) + phi_rot

                    # delta est la latitude correspondante

                    delta = np.arcsin((r*np.sin(theta))/(rho))

                    if rho <= Rp + h :

                        if begin == 0 :
                            begin = x_range

                        # Tant que le point considere est dans l'atmosphere

                        if Middle == False :

                            # Si on est au dessus ou sur la premiere inter-couche
                            if (rho - Rp) >= z_array[1] :

                                # Si r est compris entre z_step et le toit de l'atmosphere
                                if (rho - Rp) < z_array[z_array.size-1] :

                                    wh, = np.where(z_array > (rho - Rp))
                                    up = wh[0]
                                    down = wh[0]-1

                                    # Sans Middle, tous les points entre un milieu de couche et le suivant sont identifies
                                    # au niveau inter-couche qui les separe
                                    if (rho - Rp) - z_array[down] <= (z_array[down] + z_array[up])/2. :
                                        z = down
                                    else :
                                        z = up
                                else :

                                    z = z_array.size - 1

                            else :

                                # Si le point est compris entre la surface et le premier inter-couche, si r est en dessous
                                # du milieu de la premiere couche, alors il est identifie a la surface, sinon, a la
                                # premiere inter-couche
                                if (rho - Rp) <= z_array[1]/2. :
                                    z = 0
                                else :
                                    z = 1

                        else :

                            # z_array donne l'echelle d'altitude et les niveaux inter-couches, sur une maille Middle
                            # l'indice 0 correspond a la surface et les autres aux milieux de couche, en cherchant
                            # l'indice pour lesquel z_array est superieur a r, on identifie directement l'indice
                            # correspondant dans la maille Middle
                            if (rho - Rp) < h :

                                wh, = np.where(z_array > (rho - Rp))
                                z = wh[0]

                            else :

                                z = z_array.size - 1

                        # A partir de la longitude, on en deduit l'indice q correspondant dans la maille spherique, cet
                        # indice doit evoluer entre 0 (alpha -pi) et reso_long (alpha pi), sachant que le premier et le
                        # dernier point sont identiques

                        if alpha >= -np.pi/2. and alpha <= np.pi :
                            alpha += np.pi/2.
                        else :
                            alpha += 5*np.pi/2.

                        delta_o = np.arcsin(np.sin(delta)*np.cos(phi_obli)+np.cos(delta)*np.sin(phi_obli)*np.sin(alpha))
                        #alpha_o = math.atan2((-np.sin(delta)*np.sin(phi_obli)+np.cos(delta)*np.cos(phi_obli)*np.sin(alpha)),\
                        #                                    (np.cos(delta)*np.cos(alpha)))
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
                                    if x_range == begin+1 :
                                        refrac = 1

                            if phi_obli == np.pi/2. :

                                if theta_range <= theta_number/4. :
                                    alpha_o = 2*np.pi - alpha_o

                            if phi_obli == - np.pi/2. :

                                if theta_range > 3*theta_number/4. :
                                    alpha_o = 2*np.pi - alpha_o

                            if np.abs(phi_obli) > np.pi/2. :

                                if alpha_o > alpha_o_ref :
                                    alpha_o_ref = alpha_o
                                elif alpha_o == alpha_o_ref :
                                    if inv == 0 :
                                        alpha_o_ref = alpha_o
                                    else :
                                        alpha_o_ref = alpha_o
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    alpha_o_ref = alpha_o
                                    alpha_o = 2*np.pi - alpha_o
                                    if x_range == begin+1 :
                                        refrac = 1

                        if theta_range == theta_number/4. :

                            if phi_obli > 0 :
                                if phi_obli < np.pi/2. :
                                    if x < x_ref :
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    if phi_obli == np.pi/2. :
                                        alpha_o = 2*np.pi - alpha_o
                                    else :
                                        if x > -x_ref :
                                            alpha_o = 2*np.pi - alpha_o
                            else :
                                if phi_obli < -np.pi/2. :
                                    if x < -x_ref :
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    if phi_obli != -np.pi/2. :
                                        if x > x_ref :
                                            alpha_o = 2*np.pi - alpha_o

                        if theta_range > theta_number/4. and theta_range < 3*theta_number/4. :
                            # la longitude equatoriale est comprise entre 3pi/2 et pi/2 tandis que l'angle de reference des
                            # donnees GCM est compris entre -pi et 0

                            # l'angle calcule ne peut depasser pi, donc il faut lui specifier dans quelle tranche il se trouve

                            if np.abs(phi_obli) < np.pi/2. :

                                if alpha_o < alpha_o_ref :
                                    alpha_o_ref = alpha_o
                                    alpha_o = 2*np.pi - alpha_o
                                    if x_range == begin+1 :
                                        refrac = 1
                                elif alpha_o == alpha_o_ref :
                                    if inv == 0 :
                                        alpha_o_ref = alpha_o
                                        alpha_o = 2*np.pi - alpha_o
                                    else :
                                        alpha_o_ref = alpha_o
                                else :
                                    alpha_o_ref = alpha_o

                            if phi_obli == np.pi/2. :

                                if theta_range > theta_number/4. and theta_range < theta_number/2. :
                                    alpha_o = 2*np.pi - alpha_o

                            if phi_obli == - np.pi/2. :

                                if theta_range > theta_number/2. and theta_range <= 3*theta_number/4. :
                                    alpha_o = 2*np.pi - alpha_o

                            if np.abs(phi_obli) > np.pi/2. :

                                if alpha_o < alpha_o_ref :
                                    alpha_o_ref = alpha_o
                                    if x_range == begin+1 :
                                        refrac = 1
                                elif alpha_o == alpha_o_ref :
                                    if inv == 0 :
                                        alpha_o_ref = alpha_o
                                    else :
                                        alpha_o_ref = alpha_o
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    alpha_o_ref = alpha_o
                                    alpha_o = 2*np.pi - alpha_o

                        if theta_range == 3*theta_number/4. :

                            if phi_obli > 0 :
                                if phi_obli < np.pi/2. :
                                    if x < -x_ref :
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    if phi_obli != np.pi/2. :
                                        if x > x_ref :
                                            alpha_o = 2*np.pi - alpha_o
                            else :
                                if phi_obli < -np.pi/2. :
                                    if x < x_ref :
                                        alpha_o = 2*np.pi - alpha_o
                                else :
                                    if phi_obli == np.pi/2. :
                                        alpha_o = 2*np.pi - alpha_o
                                    else :
                                        if x > -x_ref :
                                            alpha_o = 2*np.pi - alpha_o

                        if x_range == begin :
                            alpha_o_ref_0 = alpha_o

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

                            q_grid[r_range,theta_range,x_range] = q_o

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

                                q_grid[r_range,theta_range,x_range] = q_o

                            else :

                                if x_range == begin+1 :

                                    if refrac == 1 :
                                        alpha_o_ref_0 = 2*np.pi - alpha_o_ref_0

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

                                    q_grid[r_range,theta_range,x_range] = q_o

                                    if alpha_o_ref_0%alpha_step < alpha_step/2. :
                                        alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step
                                    else :
                                        alpha_norm = alpha_o_ref_0 - alpha_o_ref_0%alpha_step + alpha_step

                                    if alpha_norm >= 0. and alpha_norm <= 3*np.pi/2. :
                                        q_o = int(reso_long/4 + int(round(alpha_norm*reso_long/(2*np.pi))))
                                    if alpha_norm > 3*np.pi/2. and alpha_norm <= 2*np.pi :
                                        q_o = int(int(round(alpha_norm*reso_long/(2*np.pi)))-3*reso_long/4)

                                    if q_o == reso_long :
                                        q_o = 0

                                    q_grid[r_range,theta_range,x_range-1] = q_o

                        if delta_o%delta_step < delta_step/2. :
                            delta_norm = delta_o - delta_o%delta_step
                            p_o = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))
                        else :
                            delta_norm = delta_o - delta_o%delta_step + delta_step
                            p_o = int(round(delta_norm*reso_lat/np.pi+reso_lat/2))

                        p_grid[r_range,theta_range,x_range] = p_o
                        z_grid[r_range,theta_range,x_range] = z

                if rank == 0 : 
	            bar.animate(r_range*theta_number+theta_range)

    return p_grid,q_grid,z_grid,n_level_rank

########################################################################################################################
########################################################################################################################

"""
    DX_CORRESPONDANCE

    Cette fonction calcul prealablement les distances dx et permet par la suite aux fonctions de transfert de rayonnement
    de retrouver plus rapidement les parametres atmospheriques (P,T,X_mol). On suppose qu'au moins la pression varie
    entre deux altitudes donnees.

"""
########################################################################################################################
########################################################################################################################


def dx_correspondance(p_grid,q_grid,z_grid,data,x_step,r_step,theta_step,Rp,g0,h,t,reso_long,reso_lat,n_lay_rank,Middle=False,\
                      Integral=True,Discret=True,Gravity=False,Ord=False) :

    rank = n_lay_rank[0]
    r_size,theta_size,x_size = np.shape(p_grid)
    number,t_size,z_size,lat_size,long_size = np.shape(data)
    # Sur le parcours d'un rayon lumineux, dx indique les distances parcourue dans chaque cellule de la maille spherique
    # qu'il traverse, order permet de retrouver l'indice des cellules traversees, dx_opt fourni une evaluation plus
    # precise de ces distances (dx donnant des distances en multiple de x_step)
    dx_init = np.ones((r_size,theta_size,x_size),dtype = 'int')*(-1)
    order_init = np.ones((6,r_size,theta_size,x_size),dtype = 'int')*(-1)
    dx_init_opt = np.ones((r_size,theta_size,x_size),dtype = 'float')*(-1)
    if Integral == True :
        pdx_init = np.ones((r_size,theta_size,x_size),dtype = 'float')*(-1)

    len_ref = 0

    if rank == 0 : 
        bar = ProgressBar(r_size*theta_size,'Correspondance for the optical path progression')

    for i in range(r_size) :

        r = (n_lay_rank[i]+0.5)*r_step

        for j in range(theta_size) :

            x = 0
            y = 0
            zone, = np.where(p_grid[i,j,:] >= 0)
            # L est la moitie de la distance totale que peux parcourir le rayon dans l'atmosphere a r et theta donne
            L = np.sqrt((Rp+h)**2 - (Rp+r)**2)
            Lmax = zone.size*x_step/2.
            # dist nous permet de localiser si le rayon a depasse ou non le terminateur
            dist = 0

            for k in zone :

                # On incremente dist de x_step sauf pour le premier indice, de cette maniere, les formules pour dist < Lmax
                # restent valables dans le cas ou soit z, soit lat, soit long au terminateur est different au terminateur
                # des pas precedents. Si ce n'est pas le cas, alors on passera automatiquement sur les formules dist > Lmax

                dist += x_step

                if dist <= Lmax :

                    mid = 0
                    mid_y = 0
                    passe = 0

                else :

                    if dist == Lmax + x_step/2. :

                        mid = 2
                        mid_y = 2
                        passe = 1
                    else :
                        passe = 0

                if k == zone[0] :

                    deb = int(zone[0])
                    z_2 = h

                else :

                    # Si z, lat ou long du pas k est different de z, lat ou long du pas precedent

                    if p_grid[i,j,k] != p_grid[i,j,k-1] or q_grid[i,j,k] != q_grid[i,j,k-1] or z_grid[i,j,k] != z_grid[i,j,k-1] or passe == 1 :

                        mess = ''

                        fin = int(k - 1)

                        dx_init[i,j,x] = fin - deb + 1

                        deb = int(k)

                        if Integral == True :

                            if dist < Lmax :

                                if z_grid[i,j,k] != z_grid[i,j,k-1] :

                                    z_1 = z_grid[i,j,k]*r_step
                                    mess += 'z'

                                    if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                        z_1_2 = (Rp+r)*(np.sin(j*theta_step)/(np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat))))-Rp
                                        mess += 'and p'

                                    else :

                                        z_1_2 = -1


                                    if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                        z_1_3 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                        mess += 'and q'

                                    else :

                                        z_1_3 = -1

                                    if z_1_2 != -1 or z_1_3 != -1 :
                                        if z_1_2 != -1 :
                                            if z_1_3 != -1 :
                                                z_ref = np.array([z_1,z_1_2,z_1_3])
                                                ind = np.zeros((3,3),dtype='int')

                                                wh, = np.where(z_ref == np.amax(z_ref))
                                                z_1 = z_ref[wh[0]]
                                                ind[wh[0],:] = np.array([0,1,1])
                                                wh, = np.where(z_ref == np.amin(z_ref))
                                                z_1_3 = z_ref[wh[0]]
                                                wh, = np.where((z_ref!=np.amax(z_ref))*(z_ref!=np.amin(z_ref)))
                                                z_1_2 = z_ref[wh[0]]
                                                ind[wh[0],:] = np.array([0,0,1])

                                                z_ref = np.array([z_1,z_1_2,z_1_3])

                                                for i_z in range(3) :

                                                    M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]
                                                    T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]

                                                    if Gravity == False :
                                                        g_1 = g0/(1+z_ref[i_z]/Rp)**2
                                                        g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_ref[i_z]/Rp))
                                                        P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_ref[i_z])))*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])
                                                    else :
                                                        g_1 = g0
                                                        P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])

                                                    if np.str(integ[0]) == 'inf' :
                                                        pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2))
                                                        print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                    else :
                                                        pdx_init[i,j,x] += integ[0]

                                                    z_2 = z_ref[i_z]

                                                    if Ord == True :
                                                        order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                        order_init[1,i,j,x] = p_grid[i,j,k-1+ind[1,i_z]]
                                                        order_init[2,i,j,x] = q_grid[i,j,k-1+ind[2,i_z]]
                                                        order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                        order_init[4,i,j,x] = k-1+ind[1,i_z]
                                                        order_init[5,i,j,x] = k-1+ind[2,i_z]

                                                    x = x + 1

                                                x = x - 1

                                            else :

                                                z_ref = np.array([z_1,z_1_2])
                                                ind = np.zeros((2,2),dtype='int')

                                                wh, = np.where(z_ref == np.amax(z_ref))
                                                z_1 = z_ref[wh[0]]
                                                ind[wh[0],:] = np.array([0,1])
                                                wh, = np.where(z_ref == np.amin(z_ref))
                                                z_1_2 = z_ref[wh[0]]

                                                z_ref = np.array([z_1,z_1_2])

                                                for i_z in range(2) :

                                                    M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]
                                                    T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]

                                                    if Gravity == False :
                                                        g_1 = g0/(1+z_ref[i_z]/Rp)**2
                                                        g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_ref[i_z]/Rp))
                                                        P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_ref[i_z])))*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])
                                                    else :
                                                        g_1 = g0
                                                        P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])

                                                    if np.str(integ[0]) == 'inf' :
                                                        pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2))
                                                        print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                    else :
                                                        pdx_init[i,j,x] += integ[0]

                                                    z_2 = z_ref[i_z]

                                                    if Ord == True :
                                                        order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                        order_init[1,i,j,x] = p_grid[i,j,k-1+ind[1,i_z]]
                                                        order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                        order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                        order_init[4,i,j,x] = k-1+ind[1,i_z]
                                                        order_init[5,i,j,x] = k-1

                                                    x = x + 1

                                                x = x - 1

                                        else :

                                            z_ref = np.array([z_1,z_1_3])
                                            ind = np.zeros((2,2),dtype='int')

                                            wh, = np.where(z_ref == np.amax(z_ref))
                                            z_1 = z_ref[wh[0]]
                                            ind[wh[0],:] = np.array([0,1])
                                            wh, = np.where(z_ref == np.amin(z_ref))
                                            z_1_3 = z_ref[wh[0]]

                                            z_ref = np.array([z_1,z_1_3])

                                            for i_z in range(2) :

                                                M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]
                                                T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]

                                                if Gravity == False :
                                                    g_1 = g0/(1+z_ref[i_z]/Rp)**2
                                                    g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_ref[i_z]/Rp))
                                                    P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_ref[i_z])))*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])
                                                else :
                                                    g_1 = g0
                                                    P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])

                                                if np.str(integ[0]) == 'inf' :
                                                    pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2))
                                                    print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                else :
                                                    pdx_init[i,j,x] += integ[0]

                                                z_2 = z_ref[i_z]

                                                if Ord == True :
                                                    order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                    order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,x] = q_grid[i,j,k-1+ind[1,i_z]]
                                                    order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                    order_init[4,i,j,x] = k-1
                                                    order_init[5,i,j,x] = k-1+ind[1,i_z]

                                                x = x + 1

                                            x = x - 1

                                    else :

                                        M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                        T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                        if Gravity == False :
                                            g_1 = g0/(1+z_1/Rp)**2
                                            g_0 = g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                        else :
                                            g_1 = g0
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                        if np.str(integ[0]) == 'inf' :
                                            pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                        else :
                                            pdx_init[i,j,x] = integ[0]

                                        z_2 = z_1

                                        if Ord == True :
                                            order_init[0,i,j,x] = z_grid[i,j,k-1]
                                            order_init[1,i,j,x] = p_grid[i,j,k-1]
                                            order_init[2,i,j,x] = q_grid[i,j,k-1]
                                            order_init[3,i,j,x] = k-1
                                            order_init[4,i,j,x] = k-1
                                            order_init[5,i,j,x] = k-1


                                else :

                                    if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                        z_1 = (Rp+r)*(np.sin(j*theta_step)/(np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat))))-Rp
                                        mess += 'p'

                                        if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                            z_1_2 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                            mess += 'and q'

                                            z_ref = np.array([z_1,z_1_2])
                                            ind = np.zeros((2,2),dtype='int')

                                            wh, = np.where(z_ref == np.amax(z_ref))
                                            z_1 = z_ref[wh[0]]
                                            ind[wh[0],:] = np.array([0,1])
                                            wh, = np.where(z_ref == np.amin(z_ref))
                                            z_1_2 = z_ref[wh[0]]

                                            z_ref = np.array([z_1,z_1_2])

                                            for i_z in range(2) :

                                                M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]
                                                T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]

                                                if Gravity == False :
                                                    g_1 = g0/(1+z_ref[i_z]/Rp)**2
                                                    g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_ref[i_z]/Rp))
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_ref[i_z])))*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])
                                                else :
                                                    g_1 = g0
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_ref[i_z]-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_ref[i_z]+z)/(np.sqrt((Rp+z_ref[i_z]+z)**2-(Rp+r)**2)),0,z_2-z_ref[i_z])

                                                if np.str(integ[0]) == 'inf' :
                                                    pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2))
                                                    print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                else :
                                                    pdx_init[i,j,x] += integ[0]

                                                z_2 = z_ref[i_z]

                                                if Ord == True :
                                                    order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                    order_init[1,i,j,x] = p_grid[i,j,k-1+ind[0,i_z]]
                                                    order_init[2,i,j,x] = q_grid[i,j,k-1+ind[1,i_z]]
                                                    order_init[3,i,j,x] = k-1
                                                    order_init[4,i,j,x] = k-1+ind[0,i_z]
                                                    order_init[5,i,j,x] = k-1+ind[1,i_z]

                                                x = x + 1

                                            x = x - 1

                                        else :

                                            M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                            T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                            if Gravity == False :
                                                g_1 = g0/(1+z_1/Rp)**2
                                                g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                            else :
                                                g_1 = g0
                                                P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                            if np.str(integ[0]) == 'inf' :
                                                pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                            else :
                                                pdx_init[i,j,x] = integ[0]

                                            z_2 = z_1

                                            if Ord == True :
                                                order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                order_init[3,i,j,x] = k-1
                                                order_init[4,i,j,x] = k-1
                                                order_init[5,i,j,x] = k-1

                                    else :

                                        z_1 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                        mess += 'q'

                                        M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                        T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                        if Gravity == False :
                                            g_1 = g0/(1+z_1/Rp)**2
                                            g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                        else :
                                            g_1 = g0
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                        if np.str(integ[0]) == 'inf' :
                                            pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                        else :
                                            pdx_init[i,j,x] = integ[0]

                                        z_2 = z_1

                                        if Ord == True :
                                            order_init[0,i,j,x] = z_grid[i,j,k-1]
                                            order_init[1,i,j,x] = p_grid[i,j,k-1]
                                            order_init[2,i,j,x] = q_grid[i,j,k-1]
                                            order_init[3,i,j,x] = k-1
                                            order_init[4,i,j,x] = k-1
                                            order_init[5,i,j,x] = k-1

                            else :

                                if mid == 2 :

                                    if p_grid[i,j,k] != p_grid[i,j,k-1] or q_grid[i,j,k] != q_grid[i,j,k-1] or z_grid[i,j,k] != z_grid[i,j,k-1] :

                                        if p_grid[i,j,k] != p_grid[i,j,k+1] or q_grid[i,j,k] != q_grid[i,j,k+1] or z_grid[i,j,k] != z_grid[i,j,k+1] :

                                            z_1 = np.sqrt((Rp+r)**2+(x_step/2.)**2) - Rp
                                            mid = 1
                                            center = 2

                                        else :

                                            z_1 = np.sqrt((Rp+r)**2+(x_step/2.)**2) - Rp
                                            mid = 1
                                            center = 1


                                    else :

                                        z_1 = r
                                        mid = 1
                                        center = 0

                                else :

                                    if mid == 1 and center != 2 :

                                        mid = 0
                                        center = 0
                                        z_1 = r

                                    if mid == 1 and center == 2 :

                                        mid = 0
                                        center = 1

                                if mid == 1 :

                                    if center == 1 or center == 2 :

                                        M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                        T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                        if Gravity == False :
                                            g_1 = g0/(1+z_1/Rp)**2
                                            g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                        else :
                                            g_1 = g0
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                        if np.str(integ[0]) == 'inf' :
                                            pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                        else :
                                            pdx_init[i,j,x] = integ[0]

                                        if Ord == True :
                                            order_init[0,i,j,x] = z_grid[i,j,k-1]
                                            order_init[1,i,j,x] = p_grid[i,j,k-1]
                                            order_init[2,i,j,x] = q_grid[i,j,k-1]
                                            order_init[3,i,j,x] = k-1
                                            order_init[4,i,j,x] = k-1
                                            order_init[5,i,j,x] = k-1

                                        x = x + 1

                                        z_2 = z_1
                                        z_1 = r

                                        T_1 = data[1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                        P_1 = data[0,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                        M_1 = data[number-1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                        if Gravity == False :
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                        else :
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                        if np.str(integ[0]) == 'inf' :
                                            pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                        else :
                                            pdx_init[i,j,x] += integ[0]

                                        if Ord == True :
                                            order_init[0,i,j,x] = z_grid[i,j,k]
                                            order_init[1,i,j,x] = p_grid[i,j,k]
                                            order_init[2,i,j,x] = q_grid[i,j,k]
                                            order_init[3,i,j,x] = k
                                            order_init[4,i,j,x] = k
                                            order_init[5,i,j,x] = k

                                        if center == 2 :

                                            x = x + 1

                                            T_1 = data[1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                            P_1 = data[0,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                            M_1 = data[number-1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                                            if Gravity == False :
                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                            else :
                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                            if np.str(integ[0]) == 'inf' :
                                                pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                            else :
                                                pdx_init[i,j,x] += integ[0]

                                            if Ord == True :
                                                order_init[0,i,j,x] = z_grid[i,j,k]
                                                order_init[1,i,j,x] = p_grid[i,j,k]
                                                order_init[2,i,j,x] = q_grid[i,j,k]
                                                order_init[3,i,j,x] = k
                                                order_init[4,i,j,x] = k
                                                order_init[5,i,j,x] = k

                                    else :

                                        M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                        T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                        if Gravity == False :
                                            g_1 = g0/(1+z_1/Rp)**2
                                            g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                        else :
                                            g_1 = g0
                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                        if np.str(integ[0]) == 'inf' :
                                            pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                        else :
                                            pdx_init[i,j,x] = integ[0]

                                        if Ord == True :
                                            order_init[0,i,j,x] = z_grid[i,j,k-1]
                                            order_init[1,i,j,x] = p_grid[i,j,k-1]
                                            order_init[2,i,j,x] = q_grid[i,j,k-1]
                                            order_init[3,i,j,x] = k-1
                                            order_init[4,i,j,x] = k-1
                                            order_init[5,i,j,x] = k-1

                                    z_1 = z_2

                                else :

                                    if center != 1 :

                                        if z_grid[i,j,k] != z_grid[i,j,k-1] :

                                            z_2 = z_grid[i,j,k-1]*r_step
                                            mess += 'z'

                                            if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                z_2_2 = (Rp+r)*(np.sin(j*theta_step)/(np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat))))-Rp
                                                mess += 'and p'

                                            else :

                                                z_2_2 = -1


                                            if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                                z_2_3 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                                mess += 'and q'

                                            else :

                                                z_2_3 = -1

                                            if z_2_2 != -1 or z_2_3 != -1 :
                                                if z_2_2 != -1 :
                                                    if z_2_3 != -1 :
                                                        z_ref = np.array([z_2,z_2_2,z_2_3])
                                                        ind = np.zeros((3,3),dtype='int')

                                                        wh, = np.where(z_ref == np.amin(z_ref))
                                                        z_2 = z_ref[wh[0]]
                                                        ind[wh[0],:] = np.array([0,1,1])
                                                        wh, = np.where(z_ref == np.amax(z_ref))
                                                        z_2_3 = z_ref[wh[0]]
                                                        wh, = np.where((z_ref!=np.amax(z_ref))*(z_ref!=np.amin(z_ref)))
                                                        z_2_2 = z_ref[wh[0]]
                                                        ind[wh[0],:] = np.array([0,0,1])

                                                        z_ref = np.array([z_2,z_2_2,z_2_3])

                                                        for i_z in range(3) :

                                                            M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]
                                                            T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]

                                                            if Gravity == False :
                                                                g_1 = g0/(1+z_1/Rp)**2
                                                                g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                                P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)
                                                            else :
                                                                g_1 = g0
                                                                P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1+ind[2,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)

                                                            if np.str(integ[0]) == 'inf' :
                                                                pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                            else :
                                                                pdx_init[i,j,x] += integ[0]

                                                            z_1 = z_ref[i_z]

                                                            if Ord == True :
                                                                order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                                order_init[1,i,j,x] = p_grid[i,j,k-1+ind[1,i_z]]
                                                                order_init[2,i,j,x] = q_grid[i,j,k-1+ind[2,i_z]]
                                                                order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                                order_init[4,i,j,x] = k-1+ind[1,i_z]
                                                                order_init[5,i,j,x] = k-1+ind[2,i_z]

                                                            x = x + 1

                                                        x = x - 1

                                                    else :

                                                        z_ref = np.array([z_2,z_2_2])
                                                        ind = np.zeros((2,2),dtype='int')

                                                        wh, = np.where(z_ref == np.amin(z_ref))
                                                        z_2 = z_ref[wh[0]]
                                                        ind[wh[0],:] = np.array([0,1])
                                                        wh, = np.where(z_ref == np.amax(z_ref))
                                                        z_2_2 = z_ref[wh[0]]

                                                        z_ref = np.array([z_2,z_2_2])

                                                        for i_z in range(2) :

                                                            M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]
                                                            T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]

                                                            if Gravity == False :
                                                                g_1 = g0/(1+z_1/Rp)**2
                                                                g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                                P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)
                                                            else :
                                                                g_1 = g0
                                                                P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1+ind[1,i_z]],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)

                                                            if np.str(integ[0]) == 'inf' :
                                                                pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                            else :
                                                                pdx_init[i,j,x] += integ[0]

                                                            z_1 = z_ref[i_z]

                                                            if Ord == True :
                                                                order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                                order_init[1,i,j,x] = p_grid[i,j,k-1+ind[1,i_z]]
                                                                order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                                order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                                order_init[4,i,j,x] = k-1+ind[1,i_z]
                                                                order_init[5,i,j,x] = k-1

                                                            x = x + 1

                                                        x = x - 1

                                                else :

                                                    z_ref = np.array([z_2,z_2_3])
                                                    ind = np.zeros((2,2),dtype='int')

                                                    wh, = np.where(z_ref == np.amin(z_ref))
                                                    z_2 = z_ref[wh[0]]
                                                    ind[wh[0],:] = np.array([0,1])
                                                    wh, = np.where(z_ref == np.amax(z_ref))
                                                    z_2_3 = z_ref[wh[0]]

                                                    z_ref = np.array([z_2,z_2_3])

                                                    for i_z in range(2) :

                                                        M_1 = data[number-1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]
                                                        T_1 = data[1,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]

                                                        if Gravity == False :
                                                            g_1 = g0/(1+z_1/Rp)**2
                                                            g_0 = g0/((1+(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                            P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)
                                                        else :
                                                            g_1 = g0
                                                            P_1 = data[0,t,z_grid[i,j,k-1+ind[0,i_z]],p_grid[i,j,k-1],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1+ind[0,i_z]]-0.5)*r_step))
                                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)

                                                        if np.str(integ[0]) == 'inf' :
                                                            pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                        else :
                                                            pdx_init[i,j,x] += integ[0]

                                                        z_1 = z_ref[i_z]

                                                        if Ord == True :
                                                            order_init[0,i,j,x] = z_grid[i,j,k-1+ind[0,i_z]]
                                                            order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                            order_init[2,i,j,x] = q_grid[i,j,k-1+ind[1,i_z]]
                                                            order_init[3,i,j,x] = k-1+ind[0,i_z]
                                                            order_init[4,i,j,x] = k-1
                                                            order_init[5,i,j,x] = k-1+ind[1,i_z]

                                                        x = x + 1

                                                    x = x - 1

                                            else :

                                                M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                                T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                                if Gravity == False :
                                                    g_1 = g0/(1+z_1/Rp)**2
                                                    g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                                else :
                                                    g_1 = g0
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                                if np.str(integ[0]) == 'inf' :
                                                    pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                    print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                else :
                                                    pdx_init[i,j,x] = integ[0]

                                                z_1 = z_2

                                                if Ord == True :
                                                    order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                    order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,x] = k-1
                                                    order_init[4,i,j,x] = k-1
                                                    order_init[5,i,j,x] = k-1


                                        else :

                                            if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                z_2 = (Rp+r)*(np.sin(j*theta_step)/(np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat))))-Rp
                                                mess += 'p'

                                                if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                                    z_2_2 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                                    mess += 'and q'

                                                    z_ref = np.array([z_2,z_2_2])
                                                    ind = np.zeros((2,2),dtype='int')

                                                    wh, = np.where(z_ref == np.amin(z_ref))
                                                    z_2 = z_ref[wh[0]]
                                                    ind[wh[0],:] = np.array([0,1])
                                                    wh, = np.where(z_ref == np.amax(z_ref))
                                                    z_2_2 = z_ref[wh[0]]

                                                    z_ref = np.array([z_2,z_2_2])

                                                    for i_z in range(2) :

                                                        M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]
                                                        T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]

                                                        if Gravity == False :
                                                            g_1 = g0/(1+z_1/Rp)**2
                                                            g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)
                                                        else :
                                                            g_1 = g0
                                                            P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1+ind[0,i_z]],q_grid[i,j,k-1+ind[1,i_z]]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                            integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_ref[i_z]-z_1)

                                                        if np.str(integ[0]) == 'inf' :
                                                            pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_ref[i_z])**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                            print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                        else :
                                                            pdx_init[i,j,x] += integ[0]

                                                        z_1 = z_ref[i_z]

                                                        if Ord == True :
                                                            order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                            order_init[1,i,j,x] = p_grid[i,j,k-1+ind[0,i_z]]
                                                            order_init[2,i,j,x] = q_grid[i,j,k-1+ind[1,i_z]]
                                                            order_init[3,i,j,x] = k-1
                                                            order_init[4,i,j,x] = k-1+ind[0,i_z]
                                                            order_init[5,i,j,x] = k-1+ind[1,i_z]

                                                        x = x + 1

                                                    x = x - 1

                                                else :

                                                    M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                                    T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                                    if Gravity == False :
                                                        g_1 = g0/(1+z_1/Rp)**2
                                                        g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                        P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                                    else :
                                                        g_1 = g0
                                                        P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                        integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                                    if np.str(integ[0]) == 'inf' :
                                                        pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                        print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                    else :
                                                        pdx_init[i,j,x] = integ[0]

                                                    z_1 = z_2

                                                    if Ord == True :
                                                        order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                        order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                        order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                        order_init[3,i,j,x] = k-1
                                                        order_init[4,i,j,x] = k-1
                                                        order_init[5,i,j,x] = k-1

                                            else :

                                                z_2 = np.sqrt(1+(np.cos(j*theta_step)/np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi))**2)*(Rp+r) - Rp
                                                mess += 'q'

                                                M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                                                T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                                                if Gravity == False :
                                                    g_1 = g0/(1+z_1/Rp)**2
                                                    g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                                                else :
                                                    g_1 = g0
                                                    P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                                    integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                                                if np.str(integ[0]) == 'inf' :
                                                    pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                                    print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                                                else :
                                                    pdx_init[i,j,x] = integ[0]

                                                z_1 = z_2

                                                if Ord == True :
                                                    order_init[0,i,j,x] = z_grid[i,j,k-1]
                                                    order_init[1,i,j,x] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,x] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,x] = k-1
                                                    order_init[4,i,j,x] = k-1
                                                    order_init[5,i,j,x] = k-1

                                    else :

                                        x = x - 1
                                        center = 0


                        if Discret == True :

                            if dist < Lmax :
                                # Comme z(k) < z(k-1), on resoud pythagore avec la distance au centre de l'exoplanete egale a
                                # Rp + z(k)*r_step et r = Rp + (i+0.5)*r_step puisque les rayons sont tires au milieu des couches

                                if z_grid[i,j,k] != z_grid[i,j,k-1] :

                                    if q_grid[i,j,k] != q_grid[i,j,k-1] and p_grid[i,j,k] != p_grid[i,j,k-1] :

                                        x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k] - i - 0.5) + r_step**2*(z_grid[i,j,k]**2 - (i+0.5)**2))
                                        x_pre_2 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)
                                        x_pre_3 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                        x_ref = np.array([x_pre_1,x_pre_2,x_pre_3])
                                        ind = np.zeros((3,3),dtype='int')

                                        max, = np.where(x_ref == np.amax(x_ref))
                                        ind[max,:] = np.array([0,1,1])
                                        min, = np.where(x_ref == np.amin(x_ref))
                                        mid, = np.where((x_ref != np.amax(x_ref))*(x_ref != np.amin(x_ref)))
                                        ind[mid,:] = np.array([0,0,1])

                                        dx_init_opt[i,j,y] = L - x_ref[max]
                                        L -= dx_init_opt[i,j,y]
                                        order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,0]]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,0]]
                                        order_init[3,i,j,y] = k-1+ind[0,0]
                                        order_init[4,i,j,y] = k-1+ind[1,0]
                                        order_init[5,i,j,y] = k-1+ind[2,0]
                                        y = y + 1

                                        dx_init_opt[i,j,y] = L - x_ref[mid]
                                        delta = L - x_ref[mid]
                                        L -= delta
                                        order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,1]]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,1]]
                                        order_init[3,i,j,y] = k-1+ind[0,1]
                                        order_init[4,i,j,y] = k-1+ind[1,1]
                                        order_init[5,i,j,y] = k-1+ind[2,1]
                                        y = y + 1

                                        dx_init_opt[i,j,y] = L - x_ref[min]
                                        delta = L - x_ref[min]
                                        L -= delta
                                        order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,2]]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,2]]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,2]]
                                        order_init[3,i,j,y] = k-1+ind[0,2]
                                        order_init[4,i,j,y] = k-1+ind[1,2]
                                        order_init[5,i,j,y] = k-1+ind[2,2]

                                    else :

                                        if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                            x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k] - i - 0.5) + r_step**2*(z_grid[i,j,k]**2 - (i+0.5)**2))
                                            x_pre_2 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                            x_ref = np.array([x_pre_1,x_pre_2])
                                            ind = np.zeros((2,2),dtype='int')

                                            max, = np.where(x_ref == np.amax(x_ref))
                                            ind[max,:] = np.array([0,1])
                                            min, = np.where(x_ref == np.amin(x_ref))

                                            dx_init_opt[i,j,y] = L - x_ref[max]
                                            L -= dx_init_opt[i,j,y]
                                            order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                            order_init[1,i,j,y] = p_grid[i,j,k-1]
                                            order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,0]]
                                            order_init[3,i,j,y] = k-1+ind[0,0]
                                            order_init[4,i,j,y] = k-1
                                            order_init[5,i,j,y] = k-1+ind[1,0]
                                            y = y + 1

                                            dx_init_opt[i,j,y] += L - x_ref[min]
                                            delta = L - x_ref[min]
                                            L -= delta
                                            order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                            order_init[1,i,j,y] = p_grid[i,j,k-1]
                                            order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,1]]
                                            order_init[3,i,j,y] = k-1+ind[0,1]
                                            order_init[4,i,j,y] = k-1
                                            order_init[5,i,j,y] = k-1+ind[1,1]

                                        else :

                                            if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k] - i - 0.5) + r_step**2*(z_grid[i,j,k]**2 - (i+0.5)**2))
                                                x_pre_2 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)

                                                x_ref = np.array([x_pre_1,x_pre_2])
                                                ind = np.zeros((2,2),dtype='int')

                                                max, = np.where(x_ref == np.amax(x_ref))
                                                ind[max,:] = np.array([0,1])
                                                min, = np.where(x_ref == np.amin(x_ref))

                                                dx_init_opt[i,j,y] = L - x_ref[max]
                                                L -= dx_init_opt[i,j,y]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,0]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                order_init[3,i,j,y] = k-1+ind[0,0]
                                                order_init[4,i,j,y] = k-1+ind[1,0]
                                                order_init[5,i,j,y] = k-1
                                                y = y + 1

                                                dx_init_opt[i,j,y] += L - x_ref[min]
                                                delta = L - x_ref[min]
                                                L -= delta
                                                order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,1]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                order_init[3,i,j,y] = k-1+ind[0,1]
                                                order_init[4,i,j,y] = k-1+ind[1,1]
                                                order_init[5,i,j,y] = k-1

                                            else :

                                                x_pre = np.sqrt(2*Rp*r_step*(z_grid[i,j,k] - i - 0.5) + r_step**2*(z_grid[i,j,k]**2 - (i+0.5)**2))

                                                dx_init_opt[i,j,y] = L - x_pre
                                                L -= dx_init_opt[i,j,y]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                order_init[3,i,j,y] = k-1
                                                order_init[4,i,j,y] = k-1
                                                order_init[5,i,j,y] = k-1

                                else :

                                    if q_grid[i,j,k] != q_grid[i,j,k-1] and p_grid[i,j,k] != p_grid[i,j,k-1] :

                                        x_pre_1 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)
                                        x_pre_2 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                        x_ref = np.array([x_pre_1,x_pre_2])
                                        ind = np.zeros((2,2),dtype='int')

                                        max, = np.where(x_ref == np.amax(x_ref))
                                        ind[max,:] = np.array([0,1])
                                        min, = np.where(x_ref == np.amin(x_ref))

                                        dx_init_opt[i,j,y] = L - x_ref[max]
                                        L -= dx_init_opt[i,j,y]
                                        order_init[0,i,j,y] = z_grid[i,j,k-1]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1+ind[0,0]]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,0]]
                                        order_init[3,i,j,y] = k-1
                                        order_init[4,i,j,y] = k-1+ind[0,0]
                                        order_init[5,i,j,y] = k-1+ind[1,0]

                                        y = y + 1

                                        dx_init_opt[i,j,y] += L - x_ref[min]
                                        delta = L - x_ref[min]
                                        L -= delta
                                        order_init[0,i,j,y] = z_grid[i,j,k-1]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1+ind[0,1]]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,1]]
                                        order_init[3,i,j,y] = k-1
                                        order_init[4,i,j,y] = k-1+ind[0,1]
                                        order_init[5,i,j,y] = k-1+ind[1,1]

                                    else :

                                        if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                            x_pre = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                            dx_init_opt[i,j,y] = L - x_pre
                                            L -= dx_init_opt[i,j,y]
                                            order_init[0,i,j,y] = z_grid[i,j,k-1]
                                            order_init[1,i,j,y] = p_grid[i,j,k-1]
                                            order_init[2,i,j,y] = q_grid[i,j,k-1]
                                            order_init[3,i,j,y] = k-1
                                            order_init[4,i,j,y] = k-1
                                            order_init[5,i,j,y] = k-1

                                        else :

                                            if p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                x_pre = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)

                                                dx_init_opt[i,j,y] = L - x_pre
                                                L -= dx_init_opt[i,j,y]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                order_init[3,i,j,y] = k-1
                                                order_init[4,i,j,y] = k-1
                                                order_init[5,i,j,y] = k-1

                            else :
                                # Lorsque le rayon a passe le terminateur, le premier changement de cellule permet de
                                # calculer la distance parcourue au sein de la cellule du terminateur, comme z(k) > z(k-1)
                                # on resoud pythagore avec Rp + z(k-1)*r_step

                                if mid_y == 2 :

                                    if p_grid[i,j,k] != p_grid[i,j,k-1] or q_grid[i,j,k] != q_grid[i,j,k-1] or z_grid[i,j,k] != z_grid[i,j,k-1] :

                                        dx_init_opt[i,j,y] = L - x_step/2.
                                        order_init[0,i,j,y] = z_grid[i,j,k-1]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1]
                                        order_init[3,i,j,y] = k-1
                                        order_init[4,i,j,y] = k-1
                                        order_init[5,i,j,y] = k-1

                                        y = y + 1

                                        dx_init_opt[i,j,y] = x_step/2.
                                        dx_init[i,j,y] = 1
                                        order_init[0,i,j,y] = z_grid[i,j,k]
                                        order_init[1,i,j,y] = p_grid[i,j,k]
                                        order_init[2,i,j,y] = q_grid[i,j,k]
                                        order_init[3,i,j,y] = k
                                        order_init[4,i,j,y] = k
                                        order_init[5,i,j,y] = k

                                        y = y + 1

                                        dx_init_opt[i,j,y] = x_step/2.
                                        dx_init[i,j,y] = 1
                                        order_init[0,i,j,y] = z_grid[i,j,k]
                                        order_init[1,i,j,y] = p_grid[i,j,k]
                                        order_init[2,i,j,y] = q_grid[i,j,k]
                                        order_init[3,i,j,y] = k
                                        order_init[4,i,j,y] = k
                                        order_init[5,i,j,y] = k

                                        mid_y = 1

                                        if p_grid[i,j,k] != p_grid[i,j,k+1] or q_grid[i,j,k] != q_grid[i,j,k+1] or z_grid[i,j,k] != z_grid[i,j,k+1] :
                                            center_y = 2
                                        else :
                                            center_y = 1

                                    else :
                                        # Au premier passage, le x_pre n'est que la moitie du parcours dans la cellule du
                                        # terminateur, donc on double x_pre
                                        dx_init_opt[i,j,y] = L
                                        order_init[0,i,j,y] = z_grid[i,j,k-1]
                                        order_init[1,i,j,y] = p_grid[i,j,k-1]
                                        order_init[2,i,j,y] = q_grid[i,j,k-1]
                                        order_init[3,i,j,y] = k-1
                                        order_init[4,i,j,y] = k-1
                                        order_init[5,i,j,y] = k-1

                                        mid_y = 1

                                        if p_grid[i,j,k] != p_grid[i,j,k+1] or q_grid[i,j,k] != q_grid[i,j,k+1] or z_grid[i,j,k] != z_grid[i,j,k+1] :
                                            center_y = 2
                                        else :
                                            center_y = 0
                                else :

                                    if mid_y == 0 :

                                        center_y = 0

                                    if mid_y == 1 :

                                        if center_y == 0 :
                                            ex = 0
                                            mid_y = 0

                                        if center_y == 1 :
                                            ex = x_step/2.
                                            mid_y = 0

                                        if center_y == 2 :
                                            y = y - 1
                                            ex = x_step/2.
                                            mid_y = 0

                                    if z_grid[i,j,k] != z_grid[i,j,k-1] and center_y != 2 :

                                        if q_grid[i,j,k] != q_grid[i,j,k-1] or p_grid[i,j,k] != p_grid[i,j,k-1] :

                                            if q_grid[i,j,k] != q_grid[i,j,k-1] and p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k-1] - i - 0.5) + r_step**2*(z_grid[i,j,k-1]**2 - (i+0.5)**2))
                                                x_pre_2 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)
                                                x_pre_3 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                                x_ref = np.array([x_pre_1,x_pre_2,x_pre_3])
                                                ind = np.zeros((3,3),dtype='int')

                                                max, = np.where(x_ref == np.amax(x_ref))
                                                min, = np.where(x_ref == np.amin(x_ref))
                                                ind[min,:] = np.array([0,1,1])
                                                mid, = np.where((x_ref != np.amax(x_ref))*(x_ref != np.amin(x_ref)))
                                                ind[mid,:] = np.array([0,0,1])

                                                dx_init_opt[i,j,y] = x_ref[min] - ex
                                                ex = x_ref[min]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,0]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,0]]
                                                order_init[3,i,j,y] = k-1+ind[0,0]
                                                order_init[4,i,j,y] = k-1+ind[1,0]
                                                order_init[5,i,j,y] = k-1+ind[2,0]
                                                y = y + 1

                                                dx_init_opt[i,j,y] += x_ref[mid] - ex
                                                ex = x_ref[mid]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,1]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,1]]
                                                order_init[3,i,j,y] = k-1+ind[0,1]
                                                order_init[4,i,j,y] = k-1+ind[1,1]
                                                order_init[5,i,j,y] = k-1+ind[2,1]
                                                y = y + 1

                                                dx_init_opt[i,j,y] += x_ref[max] - ex
                                                ex = x_ref[max]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,2]]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,2]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1+ind[2,2]]
                                                order_init[3,i,j,y] = k-1+ind[0,2]
                                                order_init[4,i,j,y] = k-1+ind[1,2]
                                                order_init[5,i,j,y] = k-1+ind[2,2]

                                            else :

                                                if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                                    x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k-1] - i - 0.5) + r_step**2*(z_grid[i,j,k-1]**2 - (i+0.5)**2))
                                                    x_pre_2 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                                    x_ref = np.array([x_pre_1,x_pre_2])
                                                    ind = np.zeros((2,2),dtype='int')

                                                    max, = np.where(x_ref == np.amax(x_ref))
                                                    min, = np.where(x_ref == np.amin(x_ref))
                                                    ind[min,:] = np.array([0,1])

                                                    dx_init_opt[i,j,y] = x_ref[min] - ex
                                                    ex = x_ref[min]
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,0]]
                                                    order_init[3,i,j,y] = k-1+ind[0,0]
                                                    order_init[4,i,j,y] = k-1
                                                    order_init[5,i,j,y] = k-1+ind[1,0]
                                                    y = y + 1

                                                    dx_init_opt[i,j,y] += x_ref[max] - ex
                                                    ex = x_ref[max]
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,1]]
                                                    order_init[3,i,j,y] = k-1+ind[0,1]
                                                    order_init[4,i,j,y] = k-1
                                                    order_init[5,i,j,y] = k-1+ind[1,1]

                                                else :

                                                    x_pre_1 = np.sqrt(2*Rp*r_step*(z_grid[i,j,k-1] - i - 0.5) + r_step**2*(z_grid[i,j,k-1]**2 - (i+0.5)**2))
                                                    x_pre_2 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)

                                                    x_ref = np.array([x_pre_1,x_pre_2])
                                                    ind = np.zeros((2,2),dtype='int')

                                                    max, = np.where(x_ref == np.amax(x_ref))
                                                    min, = np.where(x_ref == np.amin(x_ref))
                                                    ind[min,:] = np.array([0,1])

                                                    dx_init_opt[i,j,y] = x_ref[min] - ex
                                                    ex = x_ref[min]
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,0]]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,0]]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,y] = k-1+ind[0,0]
                                                    order_init[4,i,j,y] = k-1+ind[1,0]
                                                    order_init[5,i,j,y] = k-1
                                                    y = y + 1

                                                    dx_init_opt[i,j,y] += x_ref[max] - ex
                                                    ex = x_ref[max]
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1+ind[0,1]]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1+ind[1,1]]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,y] = k-1+ind[0,1]
                                                    order_init[4,i,j,y] = k-1+ind[1,1]
                                                    order_init[5,i,j,y] = k-1

                                        else :

                                            x_pre = np.sqrt(2*Rp*r_step*(z_grid[i,j,k-1] - i - 0.5) + r_step**2*(z_grid[i,j,k-1]**2 - (i+0.5)**2))

                                            dx_init_opt[i,j,y] = x_pre - ex
                                            ex = x_pre
                                            order_init[0,i,j,y] = z_grid[i,j,k-1]
                                            order_init[1,i,j,y] = p_grid[i,j,k-1]
                                            order_init[2,i,j,y] = q_grid[i,j,k-1]
                                            order_init[3,i,j,y] = k-1
                                            order_init[4,i,j,y] = k-1
                                            order_init[5,i,j,y] = k-1

                                    else :

                                        if center_y != 2 :

                                            if q_grid[i,j,k] != q_grid[i,j,k-1] and p_grid[i,j,k] != p_grid[i,j,k-1] :

                                                x_pre_1 = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)
                                                x_pre_2 = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                                x_ref = np.array([x_pre_1,x_pre_2])
                                                ind = np.zeros((2,2),dtype='int')

                                                max, = np.where(x_ref == np.amax(x_ref))
                                                min, = np.where(x_ref == np.amin(x_ref))
                                                ind[min,:] = np.array([0,1])

                                                dx_init_opt[i,j,y] = x_ref[min] - ex
                                                ex = x_ref[min]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[0,0]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,0]]
                                                order_init[3,i,j,y] = k-1
                                                order_init[4,i,j,y] = k-1+ind[0,0]
                                                order_init[5,i,j,y] = k-1+ind[1,0]
                                                y = y + 1

                                                dx_init_opt[i,j,y] += x_ref[max] - ex
                                                ex = x_ref[max]
                                                order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                order_init[1,i,j,y] = p_grid[i,j,k-1+ind[0,1]]
                                                order_init[2,i,j,y] = q_grid[i,j,k-1+ind[1,1]]
                                                order_init[3,i,j,y] = k-1
                                                order_init[4,i,j,y] = k-1+ind[0,1]
                                                order_init[5,i,j,y] = k-1+ind[1,1]

                                            else :

                                                if q_grid[i,j,k] != q_grid[i,j,k-1] :

                                                    x_pre = np.abs((Rp+(i+0.5)*r_step)*np.cos(j*theta_step)/(np.tan(((q_grid[i,j,k]+q_grid[i,j,k-1])/np.float(reso_long)-1)*np.pi)))

                                                    dx_init_opt[i,j,y] = x_pre - ex
                                                    ex = x_pre
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,y] = k-1
                                                    order_init[4,i,j,y] = k-1
                                                    order_init[5,i,j,y] = k-1

                                                else :

                                                    x_pre = np.sqrt(((np.sin(j*theta_step)/np.sin((p_grid[i,j,k]+p_grid[i,j,k-1]-reso_lat)/2.*np.pi/np.float(reso_lat)))**2 - 1)*(Rp+(i+0.5)*r_step)**2)

                                                    dx_init_opt[i,j,y] = x_pre - ex
                                                    ex = x_pre
                                                    order_init[0,i,j,y] = z_grid[i,j,k-1]
                                                    order_init[1,i,j,y] = p_grid[i,j,k-1]
                                                    order_init[2,i,j,y] = q_grid[i,j,k-1]
                                                    order_init[3,i,j,y] = k-1
                                                    order_init[4,i,j,y] = k-1
                                                    order_init[5,i,j,y] = k-1

                        y = y + 1
                        x = x + 1

                        # Les calculs sur z sont privilegies lorsque a la fois z et lat et/ou long changent entre deux pas
                        # successifs

                if k == zone[zone.size - 1] :

                    # Si le dernier point n'appartient pas a la meme cellule de la maille spherique que le precedent, nous
                    # avons alors calcule la distance parcourue dans l'autre cellule, mais pas la distance parcourue dans celle
                    # -ci, donc il faut ajouter un dernier dx

                    if p_grid[i,j,k] != p_grid[i,j,k-1] or q_grid[i,j,k] != q_grid[i,j,k-1] or z_grid[i,j,k] != z_grid[i,j,k-1] and k != zone[0]:

                        dx_init[i,j,x] = 1

                        if Integral == True :

                            z_2 = h

                            M_1 = data[number-1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]
                            T_1 = data[1,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]

                            if Gravity == False :
                                g_1 = g0/(1+z_1/Rp)**2
                                g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                            else :
                                g_1 = g0
                                P_1 = data[0,t,z_grid[i,j,k-1],p_grid[i,j,k-1],q_grid[i,j,k-1]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k-1]-0.5)*r_step))
                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                            if np.str(integ[0]) == 'inf' :
                                pdx_init[i,j,x] = P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                            else :
                                pdx_init[i,j,x] = integ[0]

                            if Ord == True :
                                order_init[0,i,j,x] = z_grid[i,j,k]
                                order_init[1,i,j,x] = p_grid[i,j,k]
                                order_init[2,i,j,x] = q_grid[i,j,k]
                                order_init[3,i,j,x] = k
                                order_init[4,i,j,x] = k
                                order_init[5,i,j,x] = k

                            #print(mess,i,j,k,z_1,z_2,integ[0])

                        if Discret == True :

                            L = np.sqrt((Rp+h)**2 - (Rp+r)**2)

                            dx_init_opt[i,j,y] = L - ex
                            order_init[0,i,j,y] = z_grid[i,j,k]
                            order_init[1,i,j,y] = p_grid[i,j,k]
                            order_init[2,i,j,y] = q_grid[i,j,k]
                            order_init[3,i,j,y] = k
                            order_init[4,i,j,y] = k
                            order_init[5,i,j,y] = k

                        # Si le dernier point appartient a la meme cellule que le precedent, nous n'avons pas encore calcule
                        # la distance parcourue dans cette cellule, elle est donc egale a Lmax moins le x_pre calcule
                        # au dernier changement de cellule

                    else :

                        fin = int(k)

                        dx_init[i,j,x] = fin - deb + 1

                        if Integral == True :

                            z_2 = h

                            M_1 = data[number-1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]
                            T_1 = data[1,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]

                            if Gravity == False :
                                g_1 = g0/(1+z_1/Rp)**2
                                g_0 = g0/((1+(z_grid[i,j,k-1]-0.5)*r_step/Rp)*(1+z_1/Rp))
                                P_1 = data[0,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]*np.exp(-M_1*g_0/(R_gp*T_1)*(z_1-(z_grid[i,j,k]-1.5)*r_step))
                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z/(1+z/(Rp+z_1)))*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)
                            else :
                                g_1 = g0
                                P_1 = data[0,t,z_grid[i,j,k],p_grid[i,j,k],q_grid[i,j,k]]*np.exp(-M_1*g_1/(R_gp*T_1)*(z_1-(z_grid[i,j,k]-1.5)*r_step))
                                integ = integrate.quad(lambda z:P_1/(R_gp*T_1)*N_A*np.exp(-M_1*g_1/(R_gp*T_1)*z)*(Rp+z_1+z)/(np.sqrt((Rp+z_1+z)**2-(Rp+r)**2)),0,z_2-z_1)

                            if np.str(integ[0]) == 'inf' :
                                pdx_init[i,j,x] += P_1/(R_gp*T_1)*N_A*(np.sqrt((Rp+z_2)**2-(Rp+r)**2) - np.sqrt((Rp+z_1)**2-(Rp+r)**2))
                                print('We did a correction in the integration cell (%i,%i,%i), with %.6e' %(i,j,k,pdx_init[i,j,x])), 'initial result', integ[0]
                            else :
                                pdx_init[i,j,x] += integ[0]

                            if Ord == True :
                                order_init[0,i,j,x] = z_grid[i,j,k]
                                order_init[1,i,j,x] = p_grid[i,j,k]
                                order_init[2,i,j,x] = q_grid[i,j,k]
                                order_init[3,i,j,x] = k
                                order_init[4,i,j,x] = k
                                order_init[5,i,j,x] = k

                            mess += 'end'

                            #print(mess,i,j,k,z_1,z_2,integ[0])

                        if Discret == True :

                            L = np.sqrt((Rp+h)**2 - (Rp+r)**2)

                            if x != 1 :
                                dx_init_opt[i,j,y] = L - ex
                            else :
                                dx_init_opt[i,j,y] = L
                            order_init[0,i,j,y] = z_grid[i,j,k]
                            order_init[1,i,j,y] = p_grid[i,j,k]
                            order_init[2,i,j,y] = q_grid[i,j,k]
                            order_init[3,i,j,y] = k
                            order_init[4,i,j,y] = k
                            order_init[5,i,j,y] = k


            # len_ref permet de redimensionner les tableaux

            len = np.where(order_init[0,i,j,:] != -1)[0].size

            if len > len_ref :

                len_ref = len

            if rank == 0 : 
	        bar.animate(i*theta_size+j)

    dx_grid = dx_init[:,:,0:len_ref]
    order_grid = order_init[:,:,:,0:len_ref]
    dx_grid_opt = dx_init_opt[:,:,0:len_ref]
    if Integral == True :
        pdx_grid = pdx_init[:,:,0:len_ref]
    else :
        pdx_grid = 0

    return dx_grid*x_step,dx_grid_opt,order_grid,pdx_grid


########################################################################################################################
########################################################################################################################

"""
    ALTITUDE_LINE_ARRAY2D

    Cette fonction genere les profils en pression, temperature et en fraction molaire pour un rayon incident qui se
    propagerait rectilignement a travers une atmosphere. Les effets de refraction ou de diffusion ne sont pas pris en
    compte dans cette fonction. Elle effectue une interpolation a partir des donnees produites par le LMDZ_GCM a la
    resolution adoptee pour la grille de transmitance. Pour ne pas alourdir l'interpolation, la fonction ne conserve
    que les donnees utiles et extrapole sur les coordonnees realistes des proprietes enregistrees.

    Elle retourne les donnees necessaires au calcul de transfert radiatif, completee par la fonction k_correlated_interp
    qui se charge de produire un tableau d'opacite a partir duquel la profondeur optique locale est estimee.

"""

########################################################################################################################
########################################################################################################################


def altitude_line_array2D_cyl_optimized_correspondance (r_line,theta_line,dx_grid,order_grid,Rp,h,P,T,h2o_vap,\
                                    r_step,x_step,lim_alt,Marker=False,Clouds=False,Cut=False) :

    zone, = np.where(order_grid >= 0)

    D = np.nansum(dx_grid[zone])

    if Marker == True :
        h2o_vap_ref = np.zeros(zone.size)

    T_ref = T[r_line,theta_line,order_grid[zone]]
    P_ref = P[r_line,theta_line,order_grid[zone]]
    d_ref = 2*np.sqrt((Rp + lim_alt*1000.)**2 - (Rp+r_line*r_step)**2)

    if Marker == True :
        h2o_vap_ref =h2o_vap[r_line,theta_line,order_grid[zone]]

    dx_ref = dx_grid[zone]

    Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

    if Cut == True :

        zero, = np.where(T_ref == 0)

        D -= 2*np.nansum(dx_grid[zero])
        h = lim_alt*1000.

    l = (np.sqrt((Rp+h)**2 - (Rp+r_line*r_step)**2)*2 - D)/2.

    # Pinter est en Pa, tandis que Cn_mol_inter est deja converti en densite moleculaire (m^-3)

    if Marker == True :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref,h2o_vap_ref

    else :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref


########################################################################################################################


def altitude_line_array1D_cyl_optimized_correspondance (r_line,theta_line,dx_grid,alt_grid,order_grid,Rp,h,P_col,T_col,\

                                h2o_vap_col,r_step,x_step,lim_alt,Marker=False) :

    zone, = np.where(dx_grid >= 0)

    D = np.nansum(dx_grid[zone])

    T_ref = T_col[alt_grid[order_grid[zone]]]
    P_ref = P_col[alt_grid[order_grid[zone]]]

    if Marker == True :
        h2o_vap_ref = h2o_vap_col[alt_grid[order_grid[zone]]]

    dx_ref = dx_grid[zone]

    zero, = np.where(T_ref == 0)
    no_zero, = np.where(T_ref != 0)

    Cn_mol_ref = np.zeros(P_ref.size)
    Cn_mol_ref[no_zero] = P_ref[no_zero]/(R_gp*T_ref[no_zero])*N_A

    if zero.size != 0 :

        D -= 2*np.nansum(dx_grid[zero])
        h = lim_alt*1000.

    l = (np.sqrt((Rp+h)**2 - (Rp+r_line*r_step)**2)*2 - D)/2.

    # Pinter est en Pa, tandis que Cn_mol_inter est deja converti en densite moleculaire (m^-3)

    if Marker == True :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref,h2o_vap_ref

    else :

        return zone,l,dx_ref,Cn_mol_ref,T_ref,P_ref

########################################################################################################################
########################################################################################################################

"""
    ATMOSPHERIC_MATRIX_EARTH

    Produit les matrices cylindriques de temperature, pression, fraction molaire, fraction massique, de concentration
    moalire et de concentration massique a la resolution adoptee par la matrice de reference.

    A ameliorer en lui permettant n'importe quelle resolution finale malgre la resolution de la matrice de reference
    initiale

"""

########################################################################################################################
########################################################################################################################

def atmospheric_matrix_3D(order,data,t,Rp,c_species,rank,Marker=False,Clouds=False,Composition=False) :

    sp,reso_t,reso_z,reso_lat,reso_long = np.shape(data)
    T_file = data[1,:,:,:,:]
    P_file = data[0,:,:,:,:]
    c_number = c_species.size

    if Clouds == True :

        if Marker == True :

            h2o_vap = data[2,:,:,:,:]
            gen_cond = data[3:3+c_number,:,:,:,:]
            num = 3+c_number

        else :

            gen_cond = data[2:2+c_number,:,:,:,:]
            num = 2+c_number

    else :

        if Marker == True :

            h2o_vap = data[2,:,:,:,:]
            num = 3

        else :

            num = 2


    if Composition == True :

        composit = data[num : sp,:,:,:,:]
    
    del data

    shape = np.shape(order)
    T = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)
    P = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)
    Cn = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)

    if Marker == True :
        Xm_h2o = np.zeros((shape[1],shape[2],shape[3]),dtype=np.float64)

    if Clouds == True :

        gen = np.zeros((c_number,shape[1],shape[2],shape[3]),dtype=np.float64)

    if Composition == True :

        compo = np.zeros((sp-num,shape[1],shape[2],shape[3]),dtype=np.float64)

    if rank == 0 : 
        bar = ProgressBar(shape[1]*shape[2],'Parametric recording')

    for i in range(shape[1]) :

        for j in range(shape[2]) :

            wh, = np.where(order[0,i,j,:] > 0)

            T[i,j,wh] = T_file[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]
            P[i,j,wh] = P_file[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]
            Cn[i,j,wh] = P[i,j,wh]/(R_gp*T[i,j,wh])*N_A

            if Marker == True :

                Xm_h2o[i,j,wh] = h2o_vap[t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

            if Clouds == True :

                gen[:,i,j,wh] = gen_cond[:,t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

            if Composition == True :

                compo[:,i,j,wh] = composit[:,t,order[0,i,j,wh],order[1,i,j,wh],order[2,i,j,wh]]

            if rank == 0 : 
	        bar.animate(i*shape[2] + j)

    if Marker == True :

        if Clouds == False :

            if Composition == True :

                return P,T,Cn,Xm_h2o,compo

            else :

                return P,T,Cn,Xm_h2o

        else :

            if Composition == True :

                return P,T,Cn,Xm_h2o,gen,compo

            else :

                return P,T,Cn,Xm_h2o,gen

    else :

        if Clouds == False :

            if Composition == True :

                return P,T,Cn,compo

            else :

                return P,T,Cn

        else :

            if Composition == True :

                return P,T,Cn,gen,compo

            else :

                return P,T,Cn,gen


########################################################################################################################


def atmospheric_matrix_init(p_grid,q_grid,z_grid,data,t,Rp,c_species,Marker=False,Clouds=False,Composition=False) :

    sp,reso_t,reso_z,reso_lat,reso_long = np.shape(data)
    T_file = data[1,:,:,:,:]
    P_file = data[0,:,:,:,:]
    c_number = c_species.size

    if Clouds == True :

        if Marker == True :

            h2o_vap = data[2,:,:,:,:]
            gen_cond = data[3:3+c_number,:,:,:,:]
            num = 3+c_number

        else :

            gen_cond = data[2:2+c_number,:,:,:,:]
            num = 2+c_number

    else :

        if Marker == True :

            h2o_vap = data[2,:,:,:,:]
            num = 3

        else :

            num = 2


    if Composition == True :

        composit = data[num : sp,:,:,:,:]

    shape = np.shape(p_grid)
    T = np.zeros(shape)
    P = np.zeros(shape)
    Cn = np.zeros(shape)

    if Marker == True :
        Xm_h2o = np.zeros(shape)

    if Clouds == True :

        gen = np.zeros((c_number,shape[0],shape[1],shape[2]))

    if Composition == True :

        compo = np.zeros((sp-num,shape[0],shape[1],shape[2]))

    bar = ProgressBar(shape[0],'Parametric recording')

    for i in range(shape[0]) :

        for j in range(shape[1]) :

            p_ref = -2
            q_ref = -2
            z_ref = -2

            for k in range(shape[2]) :

                p = p_grid[i,j,k]

                if p >= 0 :

                    q = q_grid[i,j,k]
                    z = z_grid[i,j,k]

                    if p == p_ref and q == q_ref and z == z_ref :

                        T[i,j,k] = T[i,j,k-1]
                        P[i,j,k] = P[i,j,k-1]
                        Cn[i,j,k] = Cn[i,j,k-1]

                        if Marker ==True :
                            Xm_h2o[i,j,k] = Xm_h2o[i,j,k-1]

                        if Clouds == True :
                            gen[:,i,j,k] = gen[:,i,j,k-1]

                        if Composition == True :
                            compo[:,i,j,k] = compo[:,i,j,k-1]

                    else :

                        if q < reso_long/2 :
                            qq = reso_long/2 + q
                        else :
                            qq = q - reso_long/2

                        pp = reso_lat - 1 - p

                        T[i,j,k] = T_file[t,z,pp,qq]
                        P[i,j,k] = P_file[t,z,pp,qq]
                        Cn[i,j,k] = P_file[t,z,pp,qq]/(R_gp*T_file[t,z,pp,qq])*N_A

                        if Marker == True :

                            Xm_h2o[i,j,k] = h2o_vap[t,z,pp,qq]

                        if Clouds == True :

                            gen[:,i,j,k] = gen_cond[:,t,z,pp,qq]

                        if Composition == True :

                            compo[:,i,j,k] = composit[:,t,z,pp,qq]

                        p_ref, q_ref, z_ref = p, q, z
            bar.animate(i + 1)


    if Marker == True :

        if Clouds == False :

            if Composition == True :

                return P,T,Xm_h2o,Cn,compo

            else :

                return P,T,Xm_h2o,Cn

        else :

            if Composition == True :

                return P,T,Xm_h2o,Cn,gen,compo

            else :

                return P,T,Xm_h2o,Cn,gen

    else :

        if Clouds == False :

            if Composition == True :

                return P,T,Cn,compo

            else :

                return P,T,Cn

        else :

            if Composition == True :

                return P,T,Cn,gen,compo

            else :

                return P,T,Cn,gen


########################################################################################################################


def atmospheric_matrix_1D(z_file,P_col,T_col,h2o_col) :

    z_grid = np.load("%s.npy"%(z_file))

    shape = np.shape(z_grid)
    T = np.zeros(shape)
    P = np.zeros(shape)
    Xm_h2o = np.zeros(shape)
    Cn = np.zeros(shape)

    j = 0

    for i in range(shape[0]) :

        z_ref = -1

        for k in range(shape[2]) :

            z = z_grid[i,j,k]

            if z >= 0 :

                if z == z_ref :

                    T[i,j,k] = T[i,j,k-1]
                    P[i,j,k] = P[i,j,k-1]
                    Xm_h2o[i,j,k] = Xm_h2o[i,j,k-1]
                    Cn[i,j,k] = Cn[i,j,k-1]

                else :

                    T[i,j,k] = T_col[z]
                    P[i,j,k] = P_col[z]
                    Xm_h2o[i,j,k] = h2o_col[z]
                    Cn[i,j,k] = P_col[z]/(R_gp*T_col[z])*N_A

                    z_ref = z

    for j in range(1,shape[1]) :

        T[:,j,:] = T[:,0,:]
        P[:,j,:] = P[:,0,:]
        Xm_h2o[:,j,:] = Xm_h2o[:,0,:]
        Cn[:,j,:] = Cn[:,0,:]


    return P,T,Xm_h2o


########################################################################################################################


def PTprofil1D(Rp,g0,M,P_surf,T_iso,n_species,x_ratio_species,r_step,delta_z,dim,number,Middle,Origin,Gravity) :

    data_convert = np.zeros((number,1,dim,1,1))

    data_convert[number - 1,:,:,:,:] += M
    data_convert[0,:,0,:,:] = P_surf
    data_convert[1,:,:,:,:] += T_iso
    for i in range(n_species.size) :
        data_convert[2+i,:,:,:,:] = x_ratio_species[i]

    bar = ProgressBar(dim,'Computation of the atmospheric dataset')

    for i_z in range(1,dim) :

        if Middle == False :

            z_ref = i_z*delta_z

        else :

            z_ref = (i_z - 0.5)*delta_z

        if Origin == True :

            if i_z != 1 :

                data_convert[0,0,i_z,0,0] = data_convert[0,0,i_z-1,0,0]*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0*\
                                    delta_z/(R_gp*data_convert[1,0,i_z-1,0,0])*1/((1+(z_ref-1*r_step)/Rp)*(1+z_ref/Rp)))
            else :

                data_convert[0,0,i_z,0,0] = data_convert[0,0,i_z-1,0,0]*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0*\
                                    delta_z/(2*R_gp*data_convert[1,0,i_z-1,0,0])*1/((1+(z_ref-0.5*r_step)/Rp)*(1+z_ref/Rp)))

        else :

            if Gravity == False :
                data_convert[0,0,i_z,0,0] = P_surf*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0/(R_gp*data_convert[1,0,i_z-1,0,0])*((z_ref/(1+z_ref/Rp))))
            else :
                data_convert[0,0,i_z,0,0] = P_surf*np.exp(-data_convert[number-1,0,i_z-1,0,0]*g0/(R_gp*data_convert[1,0,i_z-1,0,0])*z_ref)

        bar.animate(i_z + 1)

    list = np.array([])

    for i in range(number) :

        wh = np.where(data_convert[i] < 0)

        if len(wh[0]) != 0 :

            list = np.append(list,i)

    if list.size != 0 :

        mess = 'Dataset error, negative value encontered for axis : '

        for i in range(list.size) :

            mess += '%i, '%(list[i])

        mess += 'a correction is necessary, or Boxes failed'

        print mess

    return data_convert
