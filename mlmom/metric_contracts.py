"""One public source for metric argument capabilities."""
METRIC_OPTIONS = {
    'ndcg': {'k'}, 'map': {'k'}, 'mean_average_precision': {'k'}, 'mrr': {'k'},
    'roc_auc': {'average','multi_class','max_fpr'}, 'auc': {'average','multi_class','max_fpr'},
    'log_loss': set(), 'silhouette': {'sample_size','random_state','metric'},
    'average_precision': {'average','pos_label'}, 'precision': {'average','pos_label'},
    'recall': {'average','pos_label'}, 'f1': {'average','pos_label'},
    'f1_macro': {'average'}, 'macro_f1': {'average'}, 'f1_micro': {'average'},
    'micro_f1': {'average'}, 'f1_weighted': {'average'}, 'weighted_f1': {'average'},
    'rmse': {'multioutput'}, 'mse': {'multioutput'}, 'mae': {'multioutput'}, 'r2': {'multioutput'},
}


def metric_capabilities():
    return {metric: sorted(options) for metric,options in METRIC_OPTIONS.items()}
