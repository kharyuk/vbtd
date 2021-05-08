import os

import numpy as np

import numpy_tools

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

from sklearn.metrics import pairwise_distances

from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score


def preprocess_template(dataset):
    T, labels = dataset['data'], dataset['labels']
    return T, labels

def temporal_sscal(dataset):
    data = dataset['data'] - np.mean(dataset['data'], axis=1, keepdims=True)
    data /= np.std(data, axis=1, keepdims=True, ddof=1)
    labels = dataset['labels']
    return data, labels

def kmeans_gmm_clustering_experiment(
    datasets,
    save_results_path=None,
    preprocess=None,
    Ntrials=20,
    np_random_seed=None,
    verbose=True
):
    if np_random_seed is not None:
        np.random.seed(np_random_seed)
    random_states = np.random.randint(1, 1000, Ntrials)

    clust_algs = {
        'kmeans': KMeans,
        'GMM': lambda n_clusters, random_state: GMM(
            n_components=n_clusters,
            covariance_type='diag',
            tol=0.001,
            reg_covar=1e-06,
            max_iter=100,
            n_init=1,
            init_params='kmeans',
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=random_state,
            warm_start=False,
            verbose=0,
            verbose_interval=10
        )
    }

    result = np.zeros([len(datasets), len(clust_algs), Ntrials, 3])
    clust_alg_names = list(clust_algs.keys())
    dataset_names = list(datasets.keys())

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        if verbose:
            print(f'\t\t\t {dataset_name}')
        if (preprocess is not None) and (dataset_name in preprocess):
            T, labels = preprocess[dataset_name](datasets[dataset_name])
            T = numpy_tools.flatten_np(T)
        else:
            T = numpy_tools.flatten_np(datasets[dataset_name]['data'])
            labels = datasets[dataset_name]['labels']
        n_clusters = len(np.unique(labels))
        for k_alg in range(len(clust_algs)):
            for k_trial in range(Ntrials):
                clust_alg_name = clust_alg_names[k_alg]
                clustAlg = clust_algs[clust_alg_name](
                    n_clusters=n_clusters, random_state=random_states[k_trial]
                )
                try:
                    pred = clustAlg.fit_predict(T)
                    diverged = False
                except:
                    print(f'{clust_alg_name}: diverged')
                    diverged = True
                result[i, k_alg, k_trial, 0] = 0 if diverged else adjusted_rand_score(
                    labels,
                    pred
                )
                result[i, k_alg, k_trial, 1] = 0 if diverged else adjusted_mutual_info_score(
                    labels,
                    pred,
                    average_method='arithmetic'
                )
                result[i, k_alg, k_trial, 2] = 0 if diverged else fowlkes_mallows_score(
                    labels,
                    pred
                )
                if save_results_path is not None:
                    np.savez_compressed(
                        os.path.join(save_results_path),
                        result=result,
                        dataset_names=dataset_names,
                        random_states=random_states,
                        clust_alg_names=clust_alg_names,
                        Ntrials=Ntrials,
                        n_clusters=n_clusters
                    )
            if verbose:
                stat_min = np.min(result[i, k_alg, :, :], axis=0)
                stat_max = np.max(result[i, k_alg, :, :], axis=0)
                stat_med = np.median(result[i, k_alg, :, :], axis=0)
                stat_mean = np.mean(result[i, k_alg, :, :], axis=0)

                print(f'\t {clust_alg_name} min/mean/median/max')
                print(
                    f'ARI={stat_min[0]:.3f}/{stat_mean[0]:.3f}/{stat_med[0]:.3f}/{stat_max[0]:.3f}'
                )
                print(
                    f'AMI={stat_min[1]:.3f}/{stat_mean[1]:.3f}/{stat_med[1]:.3f}/{stat_max[1]:.3f}'
                )
                print(
                    f'FMI={stat_min[2]:.3f}/{stat_mean[2]:.3f}/{stat_med[2]:.3f}/{stat_max[2]:.3f}'
                )
    return result, dataset_names, random_states, clust_alg_names, n_clusters

def hac_clustering_experiment(
    datasets,
    save_results_path=None,
    preprocess=None,
    verbose=True,
    return_datasets=False
):

    affinity_names = [
        'l1', 'l2', 'cosine', 'canberra', 'correlation', 'rbf'
    ]
    linkage = [
        'complete', 'average'
    ]
    if return_datasets:
        updated_datasets = {}
    result = np.zeros([len(datasets), len(affinity_names), len(linkage), 3])
    dataset_names = list(datasets.keys())

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        if verbose:
            print(f'\t\t\t {dataset_name}')
        if (preprocess is not None) and (dataset_name in preprocess):
            T, labels = preprocess[dataset_name](datasets[dataset_name])
            T = numpy_tools.flatten_np(T)
        else:
            T = numpy_tools.flatten_np(datasets[dataset_name]['data'])
            labels = datasets[dataset_name]['labels']
        if return_datasets:
            updated_datasets[dataset_name] = {
                'data': T,
                'labels': labels
            }
        n_clusters = len(np.unique(labels))
        clustAlg = AgglomerativeClustering(n_clusters=n_clusters)
        for k_affinity in range(len(affinity_names)):
            current_affinity = affinity_names[k_affinity]
            if verbose:
                print(f'\t\t Affinity: {current_affinity}')
            for k_linkage in range(len(linkage)):
                current_linkage = linkage[k_linkage]
                if verbose:
                    print(f'\t Linkage: {current_linkage}')
                clustAlg.linkage = current_linkage
                if (
                    (current_affinity == 'canberra') or 
                    (current_affinity == 'correlation') or
                    (current_affinity == 'rbf')
                ):
                    clustAlg.affinity = 'precomputed'
                    if current_affinity == 'rbf':
                        D = pairwise_distances(T, metric='euclidean')
                        D = - np.exp(-D)
                    else:
                        D = pairwise_distances(T, metric=current_affinity)
                    pred = clustAlg.fit_predict(D)
                else:
                    clustAlg.affinity = current_affinity
                    pred = clustAlg.fit_predict(T)
                result[i, k_affinity, k_linkage, 0] = adjusted_rand_score(
                    labels,
                    pred
                )
                result[i, k_affinity, k_linkage, 1] = adjusted_mutual_info_score(
                    labels,
                    pred,
                    average_method='arithmetic'
                )
                result[i, k_affinity, k_linkage, 2] = fowlkes_mallows_score(
                    labels,
                    pred
                )
                if save_results_path is not None:
                    np.savez_compressed(
                        os.path.join(save_results_path),
                        result=result,
                        affinity_names=affinity_names,
                        linkage=linkage,
                        dataset_names=dataset_names,
                        n_clusters=n_clusters
                    )
                if verbose:
                    print(
                        f'ARI={result[i, k_affinity, k_linkage, 0]:.3f} '
                        f'AMI={result[i, k_affinity, k_linkage, 1]:.3f} '
                        f'FMI={result[i, k_affinity, k_linkage, 2]:.3f} '
                    )
    if return_datasets:
        return result, affinity_names, linkage, dataset_names, n_clusters, updated_datasets
    return result, affinity_names, linkage, dataset_names, n_clusters