import matplotlib.pyplot as plt
import seaborn as sns
import numpy_tools

def plot_performance_evolution(losses, scores_prior, scores_posterior, colors=None):
    if colors is None:
        colors = [
            sns.xkcd_rgb["pale red"],
            sns.xkcd_rgb["medium green"],
            sns.xkcd_rgb["denim blue"]
        ]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=100)#.set_facecolor('white')
    
    ax[0].plot(losses, color=sns.xkcd_rgb["dusty purple"], lw=2)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    keys = list(scores_prior.keys())
    for i in range(len(keys)):
        current_score_values = scores_prior[keys[i]]
        key = keys[i].upper()
        if isinstance(current_score_values[0], list):
            ax[1].plot(current_score_values[0], '--', label=f'{key} (train)', color=colors[i], lw=1)
            ax[1].plot(current_score_values[1], '-', label=f'{key} (valid)', color=colors[i], lw=1)
        else:
            ax[1].plot(current_score_values, label=key, color=colors[i], lw=1)
    
    keys = list(scores_posterior.keys())
    for i in range(len(keys)):
        current_score_values = scores_posterior[keys[i]]
        key = keys[i].upper()
        if isinstance(current_score_values[0], list):
            ax[2].plot(current_score_values[0], '--', label=f'{key} (train)', color=colors[i], lw=2)
            ax[2].plot(current_score_values[1], '-', label=f'{key} (valid)', color=colors[i], lw=2)
        else:
            ax[2].plot(current_score_values, label=key, color=colors[i], lw=2)
            
    ax[1].set_ylabel('Prior performance')
    ax[2].set_ylabel('Posterior performance')
    ax[1].set_xlabel('Epoch')
    ax[2].set_xlabel('Epoch')
    ax[1].legend(loc='lower right')
    ax[2].legend(loc='lower right')
    for i in range(len(ax)):
        ax[i].grid(alpha=0.5)
    #plt.savefig(fnm.replace('.npz', '.pdf'))
    #plt.show()
    return fig, ax

def plot_pixel_factors(mixture_model, image_shape, max_rank=1, mode=0):
    ncols = mixture_model.K
    if mixture_model.group_term is not None:
        ncols += 1
    fig, ax = plt.subplots(max_rank, ncols, figsize=(10, 5))
    for k in range(mixture_model.K):
        tmp = mixture_model.terms[k].get_sources(mode=mode)
        tmp = tmp.data.cpu().numpy()
        for i in range(max_rank):
            if i < tmp.shape[1]:
                ax[i, k].matshow(
                    numpy_tools.reshape_np(tmp[:, i], image_shape, use_batch=False),
                    cmap='bone'
                )
            ax[i, k].set_axis_off()
            #ax[i, k].set_title(f"{k}_{i}")
    if mixture_model.group_term is not None:
        tmp = mixture_model.group_term.get_sources(mode=mode)
        tmp = tmp.data.cpu().numpy()
        for i in range(max_rank):
            if i < tmp.shape[1]:
                ax[i, -1].matshow(
                    numpy_tools.reshape_np(tmp[:, i], image_shape, use_batch=False),
                    cmap='bone'
                )
            ax[i, -1].set_axis_off()
    return fig, ax
