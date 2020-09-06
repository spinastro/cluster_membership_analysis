from astropy.table import Table, vstack
import collections
import pandas as pd
from tqdm import tqdm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.linalg import sqrtm
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def load_cluster_tables(path_clusters,path_members):
    all_cl_members = Table.read(path_members, format='csv')['source_id','proba','cluster']
    all_names = []
    for n in all_cl_members['cluster']:
        all_names.append(n.strip())
    all_cl_members['Cluster_strip'] = all_names
    all_clusters = Table.read(path_clusters, format='fits')
    all_names = []
    for n in all_clusters['oc']:
        all_names.append(n.strip())
    all_clusters['Cluster_strip'] = all_names
    return all_clusters, all_cl_members

def def_results():
    results=collections.OrderedDict()
    results['model'] = []
    results['cluster'] = []
    results['ra'] = []
    results['dec'] = []
    results['plx'] = []
    results['s_plx'] = []
    results['pmra'] = []
    results['s_pmra'] = []
    results['pmdec'] = []
    results['s_pmdec'] = []
    results['r50'] = []
    results['Score'] = []
    results['Accuracy'] = []
    results['True_NM'] = []
    results['False_M'] = []
    results['False_NM'] = []
    results['True_M'] = []
    results['LogLoss'] = []
    results['RocScore'] = []
    results['N_mem_train'] = []
    results['N_non_mem_train'] = []
    results['N_gaia_mem'] = []
    results['N_all_mem'] = []
    results['N_non_conf_mem'] = []
    return results

def cluster_parameters(all_clusters,cluster_name):
    find_name = all_clusters['Cluster_strip'] == cluster_name
    ra = all_clusters['ra'][find_name].data[0]
    dec = all_clusters['dec'][find_name].data[0]
    plx = all_clusters['par'][find_name].data[0]
    s_plx = all_clusters['sigpar'][find_name].data[0]
    pmra = all_clusters['pmra'][find_name].data[0]
    s_pmra = all_clusters['sigpmra'][find_name].data[0]
    pmdec = all_clusters['pmdec'][find_name].data[0]
    s_pmdec = all_clusters['sigpmdec'][find_name].data[0]
    r50 = all_clusters['r50'][find_name].data[0]
    print(cluster_name)
    print("Plx: {} +/- {}".format(plx,s_plx))
    print("pmra: {} +/- {}".format(pmra,s_pmra))
    print("pmdec: {} +/- {}".format(pmdec,s_pmdec))
    print("----")
    cl_par = np.array([ra, dec, plx, s_plx, pmra, s_pmra, pmdec, s_pmdec,r50])
    return cl_par

def clean_database(data,cl_par,limit_sigma,all_cl_members,cluster_name):
    data['membership'] = np.zeros(len(data)).astype("int")
    data['prob_membership'] = np.zeros(len(data)).astype("float16")
    f_mem = all_cl_members['cluster'] == cluster_name
    
    for m_ind in range(len(all_cl_members['source_id'][f_mem])):
        find = data['source_id'] == all_cl_members['source_id'][f_mem][m_ind]
        data['prob_membership'][find] = all_cl_members['proba'][f_mem][m_ind]
        data['membership'][find] = 1

    dist_plx = abs(data['parallax']-cl_par[2])
    dist_pmra = abs(data['pmra']-cl_par[4])
    dist_pmdec = abs(data['pmdec']-cl_par[6])
    if cl_par[2] > 10.:
        f_sample = (dist_plx < limit_sigma * max([cl_par[2],1]) * cl_par[3]) & (dist_pmra < limit_sigma * cl_par[5]) & (dist_pmdec < limit_sigma * cl_par[7]) & (data['parallax'] > 2.)
    elif cl_par[2] > 2.:
        f_sample = (dist_plx < limit_sigma * max([cl_par[2],1]) * cl_par[3]) & (dist_pmra < limit_sigma * cl_par[5]) & (dist_pmdec < limit_sigma * cl_par[7]) & (data['parallax'] > 1.)
    else:
        f_sample = (dist_plx < limit_sigma * max([cl_par[2],1]) * cl_par[3]) & (dist_pmra < limit_sigma * cl_par[5]) & (dist_pmdec < limit_sigma * cl_par[7])

    f_clean = (data.mask['pmra'] == False) & (data.mask['pmdec'] == False) & (data['pmra_error'] < 2.) & (data['pmdec_error'] < 2.) & (data['parallax_error'] < 1.)# & (data['parallax'] > 0.1) & (data['parallax_error']/data['parallax'] < 0.2)
    data_clean = data[(f_clean & f_sample) | (data['prob_membership'] > 0)]
    data_clean['bp_rp'][data_clean.mask['bp_rp'] == True] = np.nan
    
    print("Limits plx: {} / {}".format(data_clean['parallax'].min(),data_clean['parallax'].max()))
    print("Limits pmra: {} / {}".format(data_clean['pmra'].min(),data_clean['pmra'].max()))
    print("Limits pmdec: {} / {}".format(data_clean['pmdec'].min(),data_clean['pmdec'].max()))
    print("----")
    print("All database {}".format(len(data)))
    print("All cleaned sample {}".format(len(data_clean)))
    return data_clean

def mem_selection(data_clean,mem_prob_threshold):
    features = ['ra','dec','parallax','pmra','pmdec']

    mem = (data_clean['prob_membership'] >= mem_prob_threshold) & (data_clean['parallax'] > 0.)
    data_clean['Mem_train_test'] = np.zeros(len(data_clean))
    data_clean['Mem_train_test'][mem] = 1
    X_members = data_clean[mem].to_pandas()[features]
    y_members = np.zeros(len(X_members)) + 1
    print('Members: {}'.format(len(X_members)))
    return data_clean, X_members, y_members

def non_mem_selection(data_clean,cl_par,X_members,prob_non_mem,multi_var_type,fraction_pool_NM):
    features = ['ra','dec','parallax','pmra','pmdec']
    data_clean['Non-mem_train_test'] = np.zeros(len(data_clean))
    
    if len(X_members) < 5:
        multi_var_type = 'lit'
    
    # Define multivariate normal and fiducial non-members selection pool
    if multi_var_type == 'mem':
        mean = np.array([cl_par[0],cl_par[1],cl_par[2],cl_par[4],cl_par[6]])
        sigmas = np.array([cl_par[8]/0.6745, cl_par[8]/0.6745, cl_par[3], cl_par[5], cl_par[7]])
        v0 =  np.array([pearsonr(X_members[features[0]],X_members[features[0]])[0], pearsonr(X_members[features[0]],X_members[features[1]])[0], pearsonr(X_members[features[0]],X_members[features[2]])[0], pearsonr(X_members[features[0]],X_members[features[3]])[0], pearsonr(X_members[features[0]],X_members[features[4]])[0]]) * sigmas[0]
        v1 =  np.array([pearsonr(X_members[features[1]],X_members[features[0]])[0], pearsonr(X_members[features[1]],X_members[features[1]])[0], pearsonr(X_members[features[1]],X_members[features[2]])[0], pearsonr(X_members[features[1]],X_members[features[3]])[0], pearsonr(X_members[features[1]],X_members[features[4]])[0]]) * sigmas[1]
        v2 =  np.array([pearsonr(X_members[features[2]],X_members[features[0]])[0], pearsonr(X_members[features[2]],X_members[features[1]])[0], pearsonr(X_members[features[2]],X_members[features[2]])[0], pearsonr(X_members[features[2]],X_members[features[3]])[0], pearsonr(X_members[features[2]],X_members[features[4]])[0]]) * sigmas[2]
        v3 =  np.array([pearsonr(X_members[features[3]],X_members[features[0]])[0], pearsonr(X_members[features[3]],X_members[features[1]])[0], pearsonr(X_members[features[3]],X_members[features[2]])[0], pearsonr(X_members[features[3]],X_members[features[3]])[0], pearsonr(X_members[features[3]],X_members[features[4]])[0]]) * sigmas[3]
        v4 =  np.array([pearsonr(X_members[features[4]],X_members[features[0]])[0], pearsonr(X_members[features[4]],X_members[features[1]])[0], pearsonr(X_members[features[4]],X_members[features[2]])[0], pearsonr(X_members[features[4]],X_members[features[3]])[0], pearsonr(X_members[features[4]],X_members[features[4]])[0]]) * sigmas[4]
        cov = [sigmas*v0,sigmas*v1,sigmas*v2,sigmas*v3,sigmas*v4]

    else:
        mean = np.array([cl_par[0],cl_par[1],cl_par[2],cl_par[4],cl_par[6]])
        cov = np.array([[(cl_par[8]/0.6745)**2,0,0,0,0],[0,(cl_par[8]/0.6745)**2,0,0,0],[0,0,cl_par[3]**2,0,0],[0,0,0,cl_par[5]**2,0],[0,0,0,0,cl_par[7]**2]])

    dis = multivariate_normal(mean=mean, cov=cov)
    array_features = np.array([data_clean['ra'],data_clean['dec'],data_clean['parallax'],data_clean['pmra'],data_clean['pmdec']])
    data_clean['Prob_multivariate_cluster'] = dis.pdf(array_features.T)/dis.pdf(mean)
    selection_pool = ((data_clean['Prob_multivariate_cluster'] < prob_non_mem) | (data_clean['parallax'] < 1./(1./cl_par[2] + 0.5))) & (data_clean['membership'] == 0)
    
    print('Size selection pool non-members: {}'.format(len(data_clean[selection_pool])))

    if len(data_clean[selection_pool]) < 1:
        return data_clean, [], []

    # Extract fiducial non-members from the pool
    lim_plx_low = min(data_clean['parallax'][selection_pool])
    lim_plx_up = max(data_clean['parallax'][selection_pool])
    bins = np.linspace(lim_plx_low,lim_plx_up,11)
    
        #Total number of fiducial NM in the first bin
    tot_num_NM = len(data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1])
        #Number of fiducial NM to be extracted in the first bin
    num_ext_NM = int(round(tot_num_NM*fraction_pool_NM+0.5))
    if num_ext_NM < len(X_members)/5:
        num_ext_NM = round(len(X_members)/5+0.5)
    if num_ext_NM > 2*len(X_members):
        num_ext_NM = 2*len(X_members)
        # Select fiducial NM for the train/validation set
    
    len_array = len(data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1])
    if num_ext_NM >= len_array:
        X_nonmembers = data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1]
        names_selected = data_clean['source_id'][selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1]
    else:
        rand = np.sort(np.array(random.sample(range(len_array), k =num_ext_NM)))
        X_nonmembers = data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1][rand]
        names_selected = data_clean['source_id'][selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==1][rand]
    for nam in names_selected:
        f = data_clean['source_id'] == nam
        data_clean['Non-mem_train_test'][f] = 1

    for k in range(2,11):
        tot_num_NM = len(data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k])
        num_ext_NM = int(round(tot_num_NM*fraction_pool_NM+0.5))
        if num_ext_NM < len(X_members)/5:
            num_ext_NM = round(len(X_members)/5+0.5)
        if num_ext_NM > 2*len(X_members):
            num_ext_NM = 2*len(X_members)
        
        len_array = len(data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k])
        if num_ext_NM >= len_array:
            X_nonmembers = vstack([X_nonmembers,data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k]])
            names_selected = data_clean['source_id'][selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k]
        else:
            rand = np.sort(np.array(random.sample(range(len_array), k =num_ext_NM)))
            X_nonmembers = vstack([X_nonmembers,data_clean[selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k][rand]])
            names_selected = data_clean['source_id'][selection_pool][np.digitize(data_clean['parallax'][selection_pool],bins)==k][rand]
        for nam in names_selected:
            f = data_clean['source_id'] == nam
            data_clean['Non-mem_train_test'][f] = 1

    X_nonmembers = X_nonmembers.to_pandas()[features]
    y_nonmembers = np.zeros(len(X_nonmembers))

    print('Non-members: {}'.format(len(X_nonmembers)))
    return data_clean, X_nonmembers, y_nonmembers

def build_train_test(X_nonmembers, y_nonmembers,X_members, y_members):
    X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_nonmembers, y_nonmembers, train_size=0.75, test_size=0.25, random_state=0)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_members, y_members, train_size=0.75, test_size=0.25, random_state=0)
    
    X_train = pd.concat([X_train_nm,X_train_m])
    X_test = pd.concat([X_test_nm,X_test_m])
    y_train = np.concatenate([y_train_nm,y_train_m])
    y_test = np.concatenate([y_test_nm,y_test_m])
    y_train = y_train.astype(np.integer)
    y_test = y_test.astype(np.integer)
    return X_train, y_train, X_test, y_test

def scale_features(data_clean, X_train, X_test):
    sc_mod = StandardScaler()
    #sc = RobustScaler()
    sc_mod.fit(X_train)
    X_train_std = sc_mod.transform(X_train).astype("float16")
    X_test_std = sc_mod.transform(X_test)
    features = ['ra','dec','parallax','pmra','pmdec']
    X_all_std = sc_mod.transform(data_clean[features].to_pandas())
    return X_train_std, X_test_std, X_all_std, sc_mod

def fitting(algorithm, X_train_std, y_train,X_test_std,y_test,cluster_name, path_results,name_model):
    if algorithm == 'gpc':
        kernel = 1.0 * RBF([1.0, 1.0, 1.0, 1.0, 1.0])  # for GPC
        #kernel = 1.0 * RBF(1.0)
        mod = GaussianProcessClassifier(kernel=kernel, random_state=0)
        mod.fit(X_train_std, y_train)
        print("Kernel {}".format(mod.kernel_))

    if algorithm == 'svc':
        n_gamma = 30
        n_C = 30
        edges_gamma = [0.1,20]
        edges_C = [0.1,20]
        gamma_par = np.linspace(edges_gamma[0],edges_gamma[1],n_gamma)
        C_par = np.linspace(edges_C[0],edges_C[1],n_C)
        svm = SVC(kernel='rbf', random_state=0, probability=True)
        param_grid = [{'C': C_par, 'gamma': gamma_par}]
        mod = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        mod.fit(X_train_std, y_train)

    if algorithm == 'gnb':
        mod = GaussianNB()
        mod.fit(X_train_std, y_train)

    if algorithm == 'qda':
        mod = QDA()
        mod.fit(X_train_std, y_train)

    return mod

def calculate_probabilities(data_clean,X_all_std,mod,sc_mod,n_iterations_probab=100):
    features = ['ra','dec','parallax','pmra','pmdec']
    err_features = ['ra_error','dec_error','parallax_error','pmra_error','pmdec_error']
    corr_features = [['ra_error','ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr'],
                     ['ra_dec_corr','dec_error','dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr'],
                     ['ra_parallax_corr','dec_parallax_corr','parallax_error','parallax_pmra_corr','parallax_pmdec_corr'],
                     ['ra_pmra_corr','dec_pmra_corr','parallax_pmra_corr','pmra_error','pmra_pmdec_corr'],
                     ['ra_pmdec_corr','dec_pmdec_corr','parallax_pmdec_corr','pmra_pmdec_corr','pmdec_error']]
                     
    data_clean['Prediction'] = mod.predict(X_all_std).astype("int8")
    data_clean['Probability'] = mod.predict_proba(X_all_std).T[1].astype("float16")
    
    my_means = np.array([data_clean[features[0]],data_clean[features[1]],data_clean[features[2]],data_clean[features[3]],data_clean[features[4]]]).T
    my_covs = np.array([])
    
    num_stars = len(data_clean)
    num_vars = 5
    my_covs = np.zeros((num_stars,num_vars, num_vars), dtype=float)
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                my_covs.T[i,j,:] = np.array(data_clean[err_features[i]]*data_clean[err_features[j]])
            else:
                my_covs.T[i,j,:] = np.array(data_clean[err_features[i]]*data_clean[err_features[j]]*data_clean[corr_features[i][j]])

    # Number of samples to generate for each (mean, cov) pair.
    nsamples = 1000
        
    # Compute the matrix square root of each covariance matrix.
    sqrtcovs = np.array([sqrtm(c) for c in my_covs])
    del my_covs
    
    # Generate samples from the standard multivariate normal distribution.
    dim = len(my_means[0])
    u = np.random.multivariate_normal(np.zeros(dim), np.eye(dim),size=(len(my_means), nsamples,)).astype("float32")# u has shape (len(means), nsamples, dim)

    # Transform u.
    v = np.einsum('ijk,ikl->ijl', u, sqrtcovs)
    del sqrtcovs
    del u
    m = np.expand_dims(my_means, 1)
    t = v + m
    del v
    del m

    data_clean['Probability_mean'] = np.zeros(len(data_clean)).astype("float16")
    data_clean['Probability_median'] = np.zeros(len(data_clean)).astype("float16")
    data_clean['Probability_std'] = np.zeros(len(data_clean)).astype("float16")

    for star in tqdm(range(len(data_clean))):
        random_data = pd.DataFrame(data=t[star])
        X_all_std = sc_mod.transform(random_data)
        prob = mod.predict_proba(X_all_std).T[1].astype("float16")
        data_clean['Probability_mean'][star] = np.mean(prob).astype("float16")
        data_clean['Probability_median'][star] = np.median(prob).astype("float16")
        data_clean['Probability_std'][star] = np.std(prob).astype("float16")

    del t, random_data, X_all_std, prob
    return data_clean

def calculate_probabilities_chunks(data_clean,X_all_std,mod,sc_mod,n_iterations_probab=100):
    features = ['ra','dec','parallax','pmra','pmdec']
    err_features = ['ra_error','dec_error','parallax_error','pmra_error','pmdec_error']
    corr_features = [['ra_error','ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr'],
                     ['ra_dec_corr','dec_error','dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr'],
                     ['ra_parallax_corr','dec_parallax_corr','parallax_error','parallax_pmra_corr','parallax_pmdec_corr'],
                     ['ra_pmra_corr','dec_pmra_corr','parallax_pmra_corr','pmra_error','pmra_pmdec_corr'],
                     ['ra_pmdec_corr','dec_pmdec_corr','parallax_pmdec_corr','pmra_pmdec_corr','pmdec_error']]
        
    data_clean['Prediction'] = mod.predict(X_all_std).astype("int8")
    data_clean['Probability'] = mod.predict_proba(X_all_std).T[1].astype("float16")
    data_clean['Probability_mean'] = np.zeros(len(data_clean)).astype("float16")
    data_clean['Probability_median'] = np.zeros(len(data_clean)).astype("float16")
    data_clean['Probability_std'] = np.zeros(len(data_clean)).astype("float16")
    
    len_data = len(data_clean)
    len_chunk = 50000
    n_chunks = round(len_data/len_chunk+0.5)
    print("Number chunks: {}".format(n_chunks))

    for ch in range(n_chunks):
        print("Chunk {}".format(ch))
        s = ch*len_chunk
        e = (ch+1)*len_chunk
        #e = min([(ch+1)*len_chunk,len_data])

        my_means = np.array([data_clean[features[0]][s:e],data_clean[features[1]][s:e],data_clean[features[2]][s:e],data_clean[features[3]][s:e],data_clean[features[4]][s:e]]).T
        my_covs = np.array([])

        num_stars = len(data_clean[s:e])
        num_vars = 5
        my_covs = np.zeros((num_stars,num_vars, num_vars), dtype=float)
        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    my_covs.T[i,j,:] = np.array(data_clean[err_features[i]][s:e]*data_clean[err_features[j]][s:e])
                else:
                    my_covs.T[i,j,:] = np.array(data_clean[err_features[i]][s:e]*data_clean[err_features[j]][s:e]*data_clean[corr_features[i][j]][s:e])

        # Number of samples to generate for each (mean, cov) pair.
        nsamples = 1000
            
        # Compute the matrix square root of each covariance matrix.
        sqrtcovs = np.array([sqrtm(c) for c in my_covs])
        del my_covs
            
        # Generate samples from the standard multivariate normal distribution.
        dim = len(my_means[0])
        u = np.random.multivariate_normal(np.zeros(dim), np.eye(dim),size=(len(my_means), nsamples,)).astype("float32")# u has shape (len(means), nsamples, dim)

        # Transform u.
        v = np.einsum('ijk,ikl->ijl', u, sqrtcovs)
        del sqrtcovs
        del u
        m = np.expand_dims(my_means, 1)
        t = v + m
        del v
        del m

        for star in tqdm(range(len(data_clean[s:e]))):
            random_data = pd.DataFrame(data=t[star])
            X_all_std = sc_mod.transform(random_data)
            prob = mod.predict_proba(X_all_std).T[1].astype("float16")
            data_clean['Probability_mean'][s:e][star] = np.mean(prob).astype("float16")
            data_clean['Probability_median'][s:e][star] = np.median(prob).astype("float16")
            data_clean['Probability_std'][s:e][star] = np.std(prob).astype("float16")
        del t, random_data, X_all_std, prob
    return data_clean

def calculate_metrics(data_clean,X_test_std,y_test,X_all_std,mod,path_results,name_model):
    score = mod.score(X_test_std,y_test)
    pred_test = mod.predict(X_test_std)
    accuracy_val = accuracy_score(y_test, pred_test)
    tNM, fM, fNM, tM = confusion_matrix(y_test, pred_test).ravel()
    log_loss_val = log_loss(y_test, pred_test)
    
    prob_test = mod.predict_proba(X_test_std)
    fpr, tpr, _ = roc_curve(y_test, prob_test.T[1])
    roc_score = roc_auc_score(y_test, prob_test.T[1])

    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.plot(fpr, tpr, label='ROC curve (area = {})'.format(round(roc_score,3)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    #plt.savefig(path_results+'plots_metrics/'+name_model+'_ROC_5D.pdf', format='pdf')
    #plt.clf()
    plt.show()


    print("Score {}".format(score))
    print("Accuracy {}".format(accuracy_val))
    print("Log Loss {}".format(log_loss_val))
    print("ROC AUC {}".format(roc_score))
    print("tNM {}, fM {}, fNM {}, tM {}".format(tNM, fM, fNM, tM))
    return score, accuracy_val, log_loss_val, roc_score, tNM, fM, fNM, tM



