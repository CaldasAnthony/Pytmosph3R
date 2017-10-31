from pytransfert import *
import pickle

########################################################################################################################
########################################################################################################################

"""
    PYDATAREAD

    Les routines suivantes assurent la conversion des fichiers de donnees de format texte en tableaux numpy afin de
    pouvoir exploiter pleinement le potentiel de traitement par cette bibliotheque python. Les fichiers de tableaux
    ainsi convertis sont enregistres dans un repertoire qui prendra le nom de l'exoplanete etudiee complete de l'
    attribut _files (exemple : Home/Earth_files/).

    Version : 6.0

    Date de derniere modification : 28.06.2016

    Date de derniere modification : 12.12.2016

"""


########################################################################################################################
########################################################################################################################


def k_corr_data_read(k_correlated_file,name,space,dec_first,dec_final,domain,P_dim,T_dim,Q_dim,dim_bande,dim_gauss,\
                     exception,directory,All=False,Jump=False) :

    data = open(k_correlated_file,'r')
    k_corr_data = data.readlines()

    ex_bande = np.array([],dtype='int')
    ex_gauss = np.array([],dtype='int')

    message = "We didn't take into account"

    if All == False :

        duo,ex_dim = np.shape(exception)

        for ex in range(ex_dim) :

            if exception[0,ex] == 'bande' :

                ex_bande = np.append(ex_bande,int(exception[1,ex]))

                message += " bande n%i,"%(int(exception[1,ex]))

            else :

                ex_gauss = np.append(ex_gauss ,int(exception[1,ex]))

                message += " gauss point n%i,"%(int(exception[1,ex]))

        message += " for this IR %ix%i resolution."%(dim_bande,dim_gauss)

        if ex_dim != 0 :

            print(message)

    else :

        print('There is no exception')

    k_corr_plan = np.zeros((T_dim,P_dim,Q_dim,dim_bande,dim_gauss))

    bar = ProgressBar(dim_gauss*dim_bande,'Kcorr record')

    for m in range(dim_gauss) :
        whg = ex_gauss[ex_gauss == m]
        if whg.size != 0 :
            m_coeff = m*T_dim*P_dim*Q_dim*dim_bande
            for l in range(dim_bande) :
                wh = ex_bande[ex_bande == l]
                if wh.size == 0 :
                    l_coeff = l*T_dim*P_dim*Q_dim
                    for k in range(Q_dim) :
                        k_coeff = k*T_dim*P_dim
                        for j in range(P_dim) :
                            j_coeff = j*T_dim
                            for i in range(T_dim) :

                                if Jump == False :

                                    i_data = i + j_coeff + k_coeff + l_coeff + m_coeff

                                    k_corr_plan[i,j,k,l,m] = np.float(k_corr_data[0][dec_first + i_data*(space+dec_final) :\
                                        dec_first + (i_data+1)*(space+dec_final)])

                                else :

                                    i_data = i + j_coeff + k_coeff + l_coeff + m_coeff

                                    i_line = i_data/3
                                    i_col = i_data%3

                                    k_corr_plan[i,j,k,l,m] = np.float(k_corr_data[i_line][dec_first + i_col*(space+dec_final) :\
                                        dec_first + (i_col+1)*(space+dec_final)])

            bar.animate(m*dim_bande+l+1)

    np.save("%sk_corr_%s_%s.npy"%(directory,name,domain),k_corr_plan)


########################################################################################################################


def k_corr_data_write(k_corr_data,k_correlated_file,dec_first,dec_final,P_dim,T_dim,Q_dim,dim_bande,dim_gauss) :

    data = open(k_correlated_file,'w')
    first = ''
    for df in range(dec_first) :
        first += ' '
    final = ''
    for dfl in range(dec_final) :
        final += ' '

    bar = ProgressBar(dim_gauss*dim_bande,'Kcorr record')

    for m in range(dim_gauss) :
        for l in range(dim_bande) :
            for k in range(Q_dim) :
                for j in range(P_dim) :
                    for i in range(T_dim) :

                        if i == 0 and j == 0 and k == 0 and l == 0 and m == 0 :
                            data.write(first)

                        data.write('%.16E'%(k_corr_data[i,j,k,l,m]))
                        data.write(final)

            bar.animate(m*dim_bande+l+1)


########################################################################################################################


def cross_data_read(file_path,type_ref,n_species,dim_P,dim_T,dim_bande) :

    section = np.zeros((n_species.size,dim_P,dim_T,dim_bande))

    for i in range(n_species.size) :

        data = pickle.load(open('%s%s.db'%(file_path,n_species[i])))

        section[i,:,:,:] = data[type_ref[0]]

    np.save("GJ1214b_files/bande_sample_bin.npy", data["%s"%(type_ref[1])])
    np.save("GJ1214b_files/P_sample_bin.npy", data["%s"%(type_ref[2])])
    np.save("GJ1214b_files/T_sample_bin.npy", data["%s"%(type_ref[3])])
    np.save("GJ1214b_files/crossection_bin.npy", section)


########################################################################################################################


def k_cont_data_read(k_cont_file,index_wl,index_T,space_wl,dec_wl_T,space_k,species,directory) :

    data = open(k_cont_file,'r')
    k_cont_data = data.readlines()

    dim_bande = np.float(k_cont_data[0][index_wl:index_wl+4])
    dim_bande = int(dim_bande)

    line,col = np.shape(k_cont_data)

    dim_T = int(line/(dim_bande+1))

    bande_cont = np.zeros(dim_bande)
    T_cont = np.zeros(dim_T)

    for i_T in range(dim_T) :

        T_cont[i_T] = np.float(k_cont_data[i_T*(dim_bande+1)][index_T:index_T+4])

    for i_bande in range(dim_bande) :

        bande_cont[i_bande] = np.float(k_cont_data[i_bande+1][0:space_wl])

    np.save('%sk_cont_nu_%s.npy'%(directory,species),bande_cont)
    np.save('%sT_cont_%s.npy'%(directory,species),T_cont)

    i_col = space_wl + dec_wl_T
    k_cont_plan = np.zeros((dim_T,dim_bande))

    bar = ProgressBar(dim_T*dim_bande,'K_cont record')

    for i_T in range(dim_T) :
        T_coeff = i_T*(dim_bande+1)
        for i_bande in range(dim_bande) :

            i_line = T_coeff + i_bande + 1

            k_cont_plan[i_T,i_bande] = np.float(k_cont_data[i_line][i_col:i_col + space_k])

            bar.animate(i_T*dim_bande+i_bande+1)

    np.save('%sk_cont_%s.npy'%(directory,species),k_cont_plan)


########################################################################################################################


def composition_data_read(composition_file,T_dim,P_dim,Q_dim,n_species,ratio,directory,renorm,Renorm=False) :

    data = open('%s'%(composition_file),'r')
    comp = data.readlines()

    x_species = np.zeros((n_species,P_dim,T_dim,Q_dim))

    bar = ProgressBar(P_dim*T_dim,'Composition record')

    for i_T in range(T_dim) :

        for i_P in range(P_dim) :

            for i_Q in range(Q_dim) :

                i_data = i_Q*T_dim*P_dim + i_P*T_dim + i_T + 5

                for i in range(2,n_species) :

                    x_species[i,i_P,i_T,i_Q] = np.float(comp[i_data][53+(i-2)*17:53+(i-1)*17])

                if Renorm == True :

                    for i_r in range(renorm[0].size) :

                        x_species[i_r,i_P,i_T,i_Q] = x_species[i_r,i_P,i_T,i_Q]/renorm[1,i_r]

                x_species[0,i_P,i_T,i_Q] = (1 - np.nansum(x_species[:,i_P,i_T,i_Q]))/(1+ratio)
                x_species[1,i_P,i_T,i_Q] = x_species[0,i_P,i_T,i_Q]*ratio

        bar.animate(i_T*P_dim+i_P+1)

    np.save('%sx_species_comp.npy'%(directory))


########################################################################################################################


def aerosol_data_read(aerosol_file,dec_first,space,species,name,directory) :

    data = open(aerosol_file,'r')
    aero_data = data.readlines()

    n_bande = aero_data[1][0:4]
    n_r = aero_data[1][0:5]

    Q = np.zeros((n_r,n_bande))
    bande = np.zeros(n_bande)
    radius = np.zeros(n_r)

    dec_line = 10 + n_bande/5 + n_r/5

    bar = ProgressBar(n_bande*n_r,'Aerosol data for %s record'%(species))

    for i in range(n_r) :

        for j in range(n_bande) :

            i_line = int(dec_line + int(j/5) + i*(n_bande/5 + 2))
            i_col = int((j%5)*space)

            Q[i,j] = np.float(aero_data[i_line][i_col:i_col+space])

            if i == 0 :

                i_b_line = int(5 + int(j/5))
                i_b_col = int((j%5)*space)
                bande[j] = np.float(aero_data[i_b_line][i_b_col:i_b_col+space])

            if j == 0 :

                i_r_line = int(7 + n_bande/5 + int(i/5))
                i_r_col = int((i%5)*space)
                radius[i] = np.float(aero_data[i_r_line][i_r_col:i_r_col+space])

            bar.animate(i*n_bande+j+1)

    np.save('%sQ_%s_%s.npy'%(directory,species,name),Q)
    np.save('%sbande_cloud_%s.npy'%(directory,name),bande)
    np.save('%sradius_cloud_%s.npy'%(directory,name),radius)
