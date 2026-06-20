import json
import pickle
import os
import numpy as np
from datetime import datetime

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'results'
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(classifier_name, method_name,
                 accuracy_results, auc_results,
                 roc_data):

    results_file = os.path.join(
        RESULTS_DIR, 'all_results.json')

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if method_name not in all_results:
        all_results[method_name] = {}

    all_results[method_name][classifier_name] = {
        'accuracy_mean': accuracy_results['mean'],
        'accuracy_std':  accuracy_results['std'],
        'auc_macro':     auc_results['macro'],
        'auc_per_class': auc_results['per_class'],
        'timestamp':     datetime.now().isoformat()
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    roc_file = os.path.join(
        RESULTS_DIR, 'roc_data.pkl')

    if os.path.exists(roc_file):
        with open(roc_file, 'rb') as f:
            all_roc = pickle.load(f)
    else:
        all_roc = {}

    if method_name not in all_roc:
        all_roc[method_name] = {}

    all_roc[method_name][classifier_name] = roc_data

    with open(roc_file, 'wb') as f:
        pickle.dump(all_roc, f)

    print(f"  Saved: {classifier_name} + {method_name}")


def load_all_results():
    
    results_file = os.path.join(
        RESULTS_DIR, 'all_results.json')

    if not os.path.exists(results_file):
        print("No results found yet. Run model.py first.")
        return {}

    with open(results_file, 'r') as f:
        return json.load(f)


def load_roc_data():
   
    roc_file = os.path.join(RESULTS_DIR, 'roc_data.pkl')

    if not os.path.exists(roc_file):
        print("No ROC data found yet. Run model.py first.")
        return {}

    with open(roc_file, 'rb') as f:
        return pickle.load(f)


def list_tested_combinations():
    
    results = load_all_results()

    if not results:
        print("Nothing tested yet.")
        return

    print(f"\n{'='*55}")
    print(f"  Tested Combinations So Far")
    print(f"{'='*55}")

    for method, classifiers in results.items():
        print(f"\n  {method}:")
        for clf, r in classifiers.items():
            print(f"    ✅ {clf:<20} "
                  f"Acc: {r['accuracy_mean']*100:.2f}%  "
                  f"AUC: {r['auc_macro']:.3f}")

    print(f"\n{'='*55}")
