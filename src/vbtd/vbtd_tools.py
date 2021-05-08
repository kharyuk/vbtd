import numpy as np
from sklearn.cluster import KMeans


def kmeans_seed_search(X, y, K, scoring, random_states=1000):
    if isinstance(random_states, int):
        random_states = np.arange(random_states)
    scores = []
    for i in range(len(sklearn_random_states)):
        clst = KMeans(n_clusters=K, random_state=random_states[i])

    #if False and gm_config is not None:
    #    clst.fit(reshape_np(data - np.mean(data, axis=0, keepdims=True), [Nsamples, -1]))
    #    y_pred = clst.predict(reshape_np(data, [Nsamples, -1]))
    #    y_pred_mr = clst.predict(reshape_np(data - np.mean(data, axis=0, keepdims=True), [Nsamples, -1]))  
    #else:
    clst.fit(X)
    y_pred = clst.predict(X)
    scores.append(scoring(labels_true=y, labels_pred=y_pred))
    return random_states, scores

def measure_usv_performance(
    mixture_model,
    guide,
    data,
    labels,
    scores=None
):
    assert scores is not None
    training = mixture_model.training
    if training:
        mixture_model.eval();
    my_pred_pi, my_pred_infer = mixture_model.predict(data, guide=guide)
    my_pred_pi = my_pred_pi.to('cpu')
    my_pred_infer = my_pred_infer.to('cpu')
    scores_prior = {}
    for key in scores:
        scores_prior[key] = scores[key](labels, my_pred_pi)
    scores_posterior = {}
    for key in scores:
        scores_posterior[key] = scores[key](labels, my_pred_infer)
    if training:
        mixture_model.train();
    return scores_prior, scores_posterior

def measure_ssv_performance(model, guide, data, labels, sv_indices, usv_indices, scores=None, verbose=True):
    assert scores is not None
    training = model.training
    if training:
        model.eval();
    train_pred_pi, train_pred_infer = model.predict(data[sv_indices], guide=guide)
    valid_pred_pi, valid_pred_infer = model.predict(data[usv_indices], guide=guide)
    if use_cuda:
        train_pred_pi = train_pred_pi.cpu()
        valid_pred_pi = valid_pred_pi.cpu()
        train_pred_infer = train_pred_infer.cpu()
        valid_pred_infer = valid_pred_infer.cpu()
    scores_prior = {}
    string = 'Prior:\t '
    for key in scores:
        scores_prior[key] = [
            scores[key](labels[sv_indices], train_pred_pi),
            scores[key](labels[usv_indices], valid_pred_pi)
        ]
        if verbose:
            string += f'{key}={scores_prior[key]:.3f}/{scores_prior[key]:.3f}\t'
    if verbose:
        print(string)
    scores_posterior = {}
    string = 'Posterior:\t '
    for key in scores:
        scores_posterior[key] = [
            scores[key](labels[sv_indices], train_pred_infer),
            scores[key](labels[usv_indices], valid_pred_infer)
        ]
        if verbose:
            string += f'{key}={scores_posterior[key][0]:.3f}/{scores_posterior[key][1]:.3f}\t'
    if verbose:
        print(string)
    if training:
        model.train();
    return scores_prior, scores_posterior

def get_pixel_factors(mixture_model, mode=0):
    result = []
    for k in range(mixture_model.K):
        tmp = mixture_model.terms[k].get_sources(mode=mode)
        tmp = tmp.data.cpu().numpy()
        result.append(tmp)

    if mixture_model.group_term is not None:
        tmp = mixture_model.group_term.get_sources(mode=mode)
        tmp = tmp.data.cpu().numpy()
        result.append(tmp)
    return result