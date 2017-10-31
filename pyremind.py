from pytransfert import *

from pyfunction import *

########################################################################################################################
########################################################################################################################

"""
    PYTRANSFERT

    Les fonctions ci-dessous permettent la recuperation des donnees precalculees, le controle des erreurs possibles ainsi
    que la resolution du transfert radiatif. Elles sont appelees exclusivement par pytransfert.

    Version : 6.0

    Date de derniere modification : 28.06.2016

    >> L'integration sur le chemin optique a ete introduite dans les fonctions du transfert radiatif
    >> Nous avons elimine la possibilite d'effectuer le calcul en 1D pur (soit en multipliant par deux la premiere
    moitie du parcours des rayons lumineux)
    >> Nous avons aussi supprime les problemes aux bords

    Date de derniere modification : 12.12.2016

"""

########################################################################################################################
########################################################################################################################

"""
    K_CORRELATED_INTERP_REMIND, K_CORRELATED_INTERP_BOUCLE_REMIND

    La fonction exploite les profils en temperature, pression et fraction molaire pour fournir en chaque point une valeur
    d'opacite qui depend de la temperature locale, de la pression locale, de l'abondance relative locale ainsi que de la
    bande et du point de Gauss considere. Elle effectue une interpolation lineaire par rapport a la temperature et la
    fraction molaire, ainsi qu'une interpolation lineaire par rapport au logarithme de la pression.

    En somme, elle cherhe les 8 points dans l'espace T,P,x pour une bande et un point de gauss donne, retrouve les
    opacites correspondante et en deduit une valeur locale par interpolation. Cette fonction s'applique sur des tableaux
    de donnees. Une premiere iteration permet de generer un ensemble de coefficients d'interpolation et les indices
    correspondants, il est ensuite reintroduit dans la fonction _BOUCLE pour eviter d'en reiterer les calculs.

"""

########################################################################################################################
########################################################################################################################


def k_correlated_interp_remind(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,size,P_rmd,P_array,T_rmd,T_array,\
            Continuum=False,Scattering= False,Clouds=False,Kcorr=False) :

    sh = np.shape(k_cloud_rmd)

    if Kcorr == True :
        tot_size,dim_bande,dim_gauss = np.shape(k_rmd)
        k_inter = np.zeros((size,dim_bande,dim_gauss))
    else :
        tot_size,dim_bande = np.shape(k_rmd)
        k_inter = np.zeros((size,dim_bande))

    k_cont_inter = np.zeros((size,dim_bande))
    k_sca_inter = np.zeros((size,dim_bande))
    k_cloud_inter = np.zeros((sh[0],size,dim_bande))

    for i in range(size) :

        P = P_array[i]
        T = T_array[i]

        if P != 0 :

            index = np.where((P_rmd == P)*(T_rmd == T))

            if Kcorr == True :
                k_inter[i,:,:] = k_rmd[index[0][0],:,:]
            else :
                k_inter[i,:] = k_rmd[index[0][0],:]

            if Continuum == True :
                k_cont_inter[i,:] = k_cont_rmd[index[0][0],:]

            if Scattering == True :
                k_sca_inter[i,:] = k_sca_rmd[index[0][0],:]

            if Clouds == True:

                k_cloud_inter[:,i,:] = k_cloud_rmd[:,index[0][0],:]

        else :

            if Kcorr == True :
                k_inter[i,:,:] = np.zeros((dim_bande,dim_gauss))
            else :
                k_inter[i,:] = np.zeros(dim_bande)

            if Continuum == True :
                k_cont_inter[i,:] = np.zeros(dim_bande)

            if Scattering == True :
                k_sca_inter[i,:] = np.zeros(dim_bande)

            if Clouds == True:

                k_cloud_inter[:,i,:] = np.zeros((sh[0],dim_bande))

    return k_inter,k_cont_inter,k_sca_inter,k_cloud_inter


########################################################################################################################


def k_correlated_interp_remind_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,size,P_rmd,P_array,T_rmd,T_array,Q_rmd,Q_array,\
            Continuum=False,Scattering= False,Clouds=False,Kcorr=False) :

    sh = np.shape(k_cloud_rmd)

    if Kcorr == True :
        tot_size,dim_bande,dim_gauss = np.shape(k_rmd)
        k_inter = np.zeros((size,dim_bande,dim_gauss))
    else :
        tot_size,dim_bande = np.shape(k_rmd)
        k_inter = np.zeros((size,dim_bande))

    if Continuum == True : 
        k_cont_inter = np.zeros((size,dim_bande))
    else : 
        k_cont_inter = np.array([])
    if Scattering == True : 
        k_sca_inter = np.zeros((size,dim_bande))
    else : 
        k_sca_inter = np.array([])
    if Clouds == True :
        k_cloud_inter = np.zeros((sh[0],size,dim_bande))
    else : 
        k_cloud_inter = np.array([])

    for i in range(size) :

        P = P_array[i]
        T = T_array[i]
        Q = Q_array[i]

        if P != 0 :

            index = np.where((P_rmd == P)*(T_rmd == T)*(Q_rmd == Q))

            if Kcorr == True :
                k_inter[i,:,:] = k_rmd[index[0][0],:,:]
            else :
                k_inter[i,:] = k_rmd[index[0][0],:]

            if Continuum == True :
                k_cont_inter[i,:] = k_cont_rmd[index[0][0],:]

            if Scattering == True :
                k_sca_inter[i,:] = k_sca_rmd[index[0][0],:]

            if Clouds == True:

                k_cloud_inter[:,i,:] = k_cloud_rmd[:,index[0][0],:]

        else :

            if Kcorr == True :
                k_inter[i,:,:] = np.zeros((dim_bande,dim_gauss))
            else :
                k_inter[i,:] = np.zeros(dim_bande)

            if Continuum == True :
                k_cont_inter[i,:] = np.zeros(dim_bande)

            if Scattering == True :
                k_sca_inter[i,:] = np.zeros(dim_bande)

            if Clouds == True:

                k_cloud_inter[:,i,:] = np.zeros((sh[0],dim_bande))

    return k_inter,k_cont_inter,k_sca_inter,k_cloud_inter


########################################################################################################################
########################################################################################################################


def k_correlated_interp_remind3D(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,size,P_rmd,P_array,\
                T_rmd,T_array,n_species,fail,rmind,Continuum=False,Isolated=False,Scattering=False,Clouds=False,Kcorr=False) :

    from pykcorr import Ksearcher, k_cont_interp_h2h2_integration, k_cont_interp_h2he_integration

    sh = np.shape(k_cloud_rmd)

    if Kcorr == True :
        if Isolated == False :
            tot_size,dim_bande,dim_gauss = np.shape((k_rmd))
        else :
            tot_size,dim_bande,dim_gauss = k_rmd
        k_inter = np.zeros((size,dim_bande,dim_gauss))
    else :
        if Isolated == False :
            tot_size,dim_bande = np.shape((k_rmd))
        else :
            tot_size,dim_bande = k_rmd
        k_inter = np.zeros((size,dim_bande))

    if Continuum == True : 
        k_cont_inter = np.zeros((size,dim_bande))
    else : 
        k_cont_inter = np.array([])
    if Scattering == True : 
        k_sca_inter = np.zeros((size,dim_bande))
    else : 
        k_sca_inter = np.array([])
    if Clouds == True :
        k_cloud_inter = np.zeros((sh[0],size,dim_bande))
    else : 
        k_cloud_inter = np.array([])

    # Le logarithme est applique ici a un pression en Pa, les fichiers reminds ont ete ranges par tranches de decade
    # afin d'alleger les temps de calcul de la fonction np.where(), la routine va donc determienr dans quelle decade
    # le point de pression se trouve et en deduire les opacites correspondantes

    for i in range(size) :

        P = P_array[i]

        if P == 0 :

            index = np.array([])

        else :

            T = T_array[i]
            pp = np.log10(P)

            wh, = np.where(rmind[1] > pp)
            if len(wh) == 0 :
	        if rmind[1,rmind[1].size-1] == pp :
		    ind = rmind[1].size-1
	    else : 
	        ind = wh[0]

            index, = np.where((P_rmd[np.int(rmind[0,ind-1]):np.int(rmind[0,ind])+1] == P)*(T_rmd[np.int(rmind[0,ind-1]):np.int(rmind[0,ind])+1] == T))

        if len(index) != 0 and P != 0 :

	    index = np.int(rmind[0,ind-1])+index[0]

            if Kcorr == True :
                if Isolated == False :
                    k_inter[i,:,:] = k_rmd[index,:,:]
            else :
                if Isolated == False :
                    k_inter[i,:] = k_rmd[index,:]

            if Continuum == True :
                k_cont_inter[i,:] = k_cont_rmd[index,:]
            if Scattering == True :
                k_sca_inter[i,:] = k_sca_rmd[index,:]

            if Clouds == True :
                k_cloud_inter[:,i,:] = k_cloud_rmd[:,index,:]


        else :

            if P != 0 :

                print("Failure of the remind routine for couple (%s, %s)" %(P,T))

                fail += 1

                # Si le couple P,T,x n'est pas reconnu, la routine a pour instruction d'indiquer le couple en question
                # et d'affecter l'opacite de la cellule precedente (etant donne qu'elles sont balayee le long des rayons
                # les couples successifs sont tres proches les uns des autres

                if Kcorr == True :
                    if Isolated == True :
                        k_inter[i,:,:] = k_inter[i-1,:,:]
                else :
                    if Isolated == True :
                        k_inter[i,:] = k_inter[i-1,:]

                if Continuum == True :
                    k_cont_inter[i,:] = k_cont_inter[i-1,:]
                if Scattering == True :
                    k_sca_inter[i,:] = k_sca_inter[i-1,:]

                if Clouds == True :
                    k_cloud_inter[:,i,:] = k_cloud_inter[:,i-1,:]

    return k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail


########################################################################################################################
########################################################################################################################


def k_correlated_interp_remind3D_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,size,P_rmd,P_array,\
                T_rmd,T_array,Q_rmd,Q_array,n_species,fail,rmind,Continuum=False,Isolated=False,Scattering=False,Clouds=False,Kcorr=False) :

    from pykcorr import Ksearcher, k_cont_interp_h2h2_integration, k_cont_interp_h2he_integration

    sh = np.shape(k_cloud_rmd)

    if Kcorr == True :
        if Isolated == False :
            tot_size,dim_bande,dim_gauss = np.shape((k_rmd))
        else :
            tot_size,dim_bande,dim_gauss = k_rmd
        k_inter = np.zeros((size,dim_bande,dim_gauss))
    else :
        if Isolated == False :
            tot_size,dim_bande = np.shape((k_rmd))
        else :
            tot_size,dim_bande = k_rmd
        k_inter = np.zeros((size,dim_bande))

    k_cont_inter = np.zeros((size,dim_bande))
    k_sca_inter = np.zeros((size,dim_bande))
    k_cloud_inter = np.zeros((sh[0],size,dim_bande))

    # Le logarithme est applique ici a un pression en Pa, les fichiers reminds ont ete ranges par tranches de decade
    # afin d'alleger les temps de calcul de la fonction np.where(), la routine va donc determienr dans quelle decade
    # le point de pression se trouve et en deduire les opacites correspondantes

    for i in range(size) :

        P = P_array[i]

        if P == 0 :

            index = np.array([])

        else :

            T = T_array[i]
            Q = Q_array[i]
            pp = np.log10(P)

            wh, = np.where(rmind[1] > pp)
            ind = wh[0]

            index, = np.where((P_rmd[np.int(rmind[0,ind-1]):np.int(rmind[0,ind])+1] == P)*(T_rmd[np.int(rmind[0,ind-1]):np.int(rmind[0,ind])+1] == T)*(Q_rmd[np.int(rmind[0,ind-1]):np.int(rmind[0,ind])+1] == Q))

        if len(index) != 0 and P != 0 :

            index = np.int(rmind[0,ind-1])+index[0]

            if Kcorr == True :
                if Isolated == False :
                    k_inter[i,:,:] = k_rmd[index,:,:]
            else :
                if Isolated == False :
                    k_inter[i,:] = k_rmd[index,:]

            if Continuum == True :
                k_cont_inter[i,:] = k_cont_rmd[index,:]
            if Scattering == True :
                k_sca_inter[i,:] = k_sca_rmd[index,:]

            if Clouds == True :
                k_cloud_inter[:,i,:] = k_cloud_rmd[:,index,:]


        else :

            if P != 0 :

                print("Failure of the remind routine for couple (%s, %s)" %(np.log10(P),T))

                fail += 1

                # Si le couple P,T,x n'est pas reconnu, la routine a pour instruction d'indiquer le couple en question
                # et d'affecter l'opacite de la cellule precedente (etant donne qu'elles sont balayee le long des rayons
                # les couples successifs sont tres proches les uns des autres

                if Kcorr == True :
                    if Isolated == True :
                        k_inter[i,:,:] = k_inter[i-1,:,:]
                else :
                    if Isolated == True :
                        k_inter[i,:] = k_inter[i-1,:]

                if Continuum == True :
                    k_cont_inter[i,:] = k_cont_inter[i-1,:]
                if Scattering == True :
                    k_sca_inter[i,:] = k_sca_inter[i-1,:]

                if Clouds == True :
                    k_cloud_inter[:,i,:] = k_cloud_inter[:,i-1,:]

    return k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail


########################################################################################################################
########################################################################################################################

"""
    RADIATIVE_TRANSFERT_REMIND, RADIATIVE_TRANSFERT_REMIND_3D

    Ces fonctions sont les plus importantes car elles effectuent les calculs de profondeur optique, puis de transmittance.
    Elles utilisent les lignes en couples P,T,x preetablies ainsi que les opacites de telle sorte que l'ensemble des
    calculs peuvent etre transformes en calculs matriciels. Les differentes options permettent quant a elles de separer
    les contributions des nuages, des collisions, de la diffusion Rayleigh ou l'absorption moleculaire. La routine s'
    adapte aussi bien aux k-correles qu'au raie par raie du moment ou cela est bien specifie.

    Dans le cas des k-correles, les calculs s'arrete du moment ou la profondeur optique devient tellement grande, ou
    tellement petite que la contribution des derniers points de Gauss n'a plus besoin d'etre calculee, allegeant ainsi
    les calculs. Le critere adopte est neanmoins arbitraire.

"""


########################################################################################################################
########################################################################################################################


def radiative_transfert_remind3D(dx_ref,pdx_ref,Cn_ref,K,K_cont,K_sca,K_cloud,gauss_val,single,continuum=False,\
                                 isolated=False,scattering=False,clouds=False,kcorr=False,integral=False) :

    if kcorr == True :

        size, dim_bande,dim_gauss = np.shape(K)

        if clouds == True :
            sh = np.shape(K_cloud)

        g, = np.where(dx_ref > 0.)

        dx_ref = dx_ref[g]
        Cn_ref = Cn_ref[g]
        if integral == True :
            pdx_ref = pdx_ref[g]

        t_cont = 0
        t_sca = 0
        t_cloud = 0

        if continuum == True :

            k_cont = K_cont[g,:]

        if scattering == True :

            k_sca = K_sca[g,:]

        if clouds == True :

            k_cloud = K_cloud[:,g,:]

        k_inter = K[g,:,:]

        if isolated == False :

            t = np.zeros((dim_bande,dim_gauss))
            tau = np.zeros((dim_bande,dim_gauss))

            if integral == True :

                for i_bande in range(dim_bande) :

                    t[i_bande,:] = np.dot(pdx_ref,k_inter[:,i_bande,:])

            else :

                for i_bande in range(dim_bande) :

                    t[i_bande,:] = np.dot(Cn_ref*dx_ref,k_inter[:,i_bande,:])

            if scattering == True :

                t_sca = np.dot(dx_ref,k_sca)

            if continuum == True :

                t_cont = np.dot(Cn_ref*dx_ref,k_cont)

            if clouds == True :

                if single == "no" :

                    for c_num in range(sh[0]) :

                        t_cloud += np.dot(dx_ref,k_cloud[c_num,:,:])

                else :

                    t_cloud = np.dot(dx_ref,k_cloud[single,:])

            for i_gauss in range(dim_gauss) :

                tau[:,i_gauss] = -(t[:,i_gauss]+t_cont+t_sca+t_cloud)

            I_out = np.dot(np.exp(tau),gauss_val)

        else :

            if scattering == True :

                t_sca = np.dot(dx_ref,k_sca)

            if continuum == True :

                t_cont = np.dot(Cn_ref*dx_ref,k_cont)

            if clouds == True :

                if single == "no" :

                    for c_num in range(sh[0]) :

                        t_cloud += np.dot(dx_ref,k_cloud[c_num,:,:])

                else :

                    t_cloud = np.dot(dx_ref,k_cloud[single,:])

            tau = -(t_cont+t_sca+t_cloud)

            I_out = np.exp(tau)

    else :

        if clouds == True :
            sh = np.shape(K_cloud)

        g, = np.where(Cn_ref > 0.)

        t_cont = 0
        t_sca = 0
        t_cloud = 0

        dx_ref = dx_ref[g]
        Cn_ref = Cn_ref[g]
        if integral == True :
            pdx_ref = pdx_ref[g]

        if isolated == False :

            k_inter = K[g,:]

            if integral == True :

                t = np.dot(pdx_ref,k_inter)

            else :

                t = np.dot(Cn_ref*dx_ref,k_inter)

            if scattering == True :

                k_sca = K_sca[g,:]

                t_sca = np.dot(dx_ref,k_sca)

            if continuum == True :

                k_cont = K_cont[g,:]

                t_cont = np.dot(Cn_ref*dx_ref,k_cont)

            if clouds == True :

                k_cloud = K_cloud[:,g,:]

                if single == "no" :

                    for c_num in range(sh[0]) :

                        t_cloud += np.dot(dx_ref,k_cloud[c_num,:,:])

                else :

                    t_cloud += np.dot(dx_ref,k_cloud[single,:,:])

            I_out = np.exp(-(t+t_cont+t_sca+t_cloud))

        else :

            if continuum == True :

                k_cont = K_cont[g,:]

                t_cont = np.dot(Cn_ref*dx_ref,k_cont)

            if scattering == True :

                k_sca = K_sca[g,:]

                t_sca = np.dot(dx_ref,k_sca)

            if clouds == True :

                k_cloud = K_cloud[:,g,:]

                if single == "no" :

                    for c_num in range(sh[0]) :

                        t_cloud += np.dot(dx_ref,k_cloud[c_num,:,:])

                else :

                    t_cloud += np.dot(dx_ref,k_cloud[single,:,:])

            I_out = np.exp(-(t_cont+t_sca+t_cloud))

    return I_out
