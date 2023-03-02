from dal_toolbox.metrics.generalization import area_under_curve

def final_results(results):    
    acc = [cycles['test_stats']['test_acc'] for cycles in results.values()]
    f1_macro = [cycles['test_stats']['test_f1_macro'] for cycles in results.values()]
    f1_micro = [cycles['test_stats']['test_f1_micro'] for cycles in results.values()]
    acc_blc = [cycles['test_stats']['test_acc_blc'] for cycles in results.values()]

    auc_acc = area_under_curve(acc)
    auc_f1_macro = area_under_curve(f1_macro)
    auc_f1_micro = area_under_curve(f1_micro)
    auc_acc_blc = area_under_curve(acc_blc)

    auc_results = {
        'final_auc_acc': auc_acc,
        'final_auc_f1_macro': auc_f1_macro,
        'final_auc_f1_micro': auc_f1_micro,
        'final_auc_acc_blc': auc_acc_blc 
    }

    return auc_results