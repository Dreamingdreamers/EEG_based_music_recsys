import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_all_subjects_cached
from results_storage import save_results, list_tested_combinations

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier,
                              RandomForestClassifier,
                              ExtraTreesClassifier,
                              BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   label_binarize)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold,
                                     cross_val_score,
                                     train_test_split)
from sklearn.metrics import roc_auc_score, roc_curve, auc

def apply_balancing(X, y, method='smote'):
    from imblearn.over_sampling import (SMOTE,
                                        BorderlineSMOTE,
                                        ADASYN)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Applying {method.upper()}...")

    try:
        if method == 'smote':
            sampler = SMOTE(k_neighbors=5, random_state=42)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(
                k_neighbors=5, random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(
                n_neighbors=5, random_state=42)

        X_res, y_res = sampler.fit_resample(X, y_enc)
        y_res = le.inverse_transform(y_res)
        print(f"Done: {len(y_res)} samples")
        return X_res, y_res

    except Exception as e:
        print(f"{method} failed ({e}), using SMOTE fallback...")
        sampler = SMOTE(k_neighbors=5, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y_enc)
        y_res = le.inverse_transform(y_res)
        return X_res, y_res

def test_one_classifier(classifier_name, pipeline,
                        X, y, method_name):
    """
    Test a single classifier with given balanced data.
    Computes CV accuracy, AUC-ROC, and ROC curve data.
    Saves results to disk automatically.
    """
    print(f"\n{'═'*55}")
    print(f"  Testing: {classifier_name} + {method_name}")
    print(f"{'═'*55}")

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y_encoded,
        cv=cv, scoring='accuracy', n_jobs=-1)

    accuracy_results = {
        'mean': scores.mean(),
        'std':  scores.std()
    }
    print(f"CV Accuracy: {scores.mean()*100:.2f}% "
          f"(±{scores.std()*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2,
        stratify=y_encoded, random_state=42)

    n_classes  = len(le.classes_)
    y_test_bin = label_binarize(
        y_test, classes=range(n_classes))

    try:
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)
        has_proba = True
    except AttributeError:
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.decision_function(X_test)
        if y_prob.ndim == 1:
            y_prob = np.column_stack(
                [-y_prob, y_prob])
        has_proba = False
        print("  (using decision_function, no predict_proba)")

    class_aucs = {}
    fpr_dict, tpr_dict = {}, {}

    for i, cls in enumerate(le.classes_):
        try:
            fpr, tpr, _ = roc_curve(
                y_test_bin[:, i], y_prob[:, i])
            class_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, class_auc = [0, 1], [0, 1], 0.5
        class_aucs[cls] = class_auc
        fpr_dict[cls]   = list(fpr)
        tpr_dict[cls]   = list(tpr)

    try:
        macro_auc = roc_auc_score(
            y_test_bin, y_prob,
            multi_class='ovr', average='macro')
    except Exception:
        macro_auc = np.mean(list(class_aucs.values()))

    auc_results = {
        'macro':     macro_auc,
        'per_class': class_aucs
    }
    print(f"Macro AUC-ROC: {macro_auc:.4f}")

    roc_data = {
        'fpr': fpr_dict,
        'tpr': tpr_dict,
        'auc': class_aucs
    }

    save_results(
        classifier_name, method_name,
        accuracy_results, auc_results, roc_data)

    return accuracy_results, auc_results

if __name__ == "__main__":
    print("=" * 55)
    print("  ThetaPlay — Classic + Ensemble Classifiers")
    print("=" * 55)

    print("\nLoading data...")
    X_raw, y_raw = load_all_subjects_cached()

    TODAYS_CLASSIFIERS = {

        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=42))
        ]),

        'Naive Bayes': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),

        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))
        ]),

        'AdaBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42))
        ]),

        'Extra Trees': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))
        ]),

        'Bagging': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=15,
                    class_weight='balanced'),
                n_estimators=100,
                random_state=42,
                n_jobs=-1))
        ]),

        'Ridge Classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RidgeClassifier(
                alpha=1.0,
                class_weight='balanced',
                random_state=42))
        ]),

        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis(
                solver='svd'))
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))
        ]),

    } 

    METHODS_TODAY = ['smote', 'borderline', 'adasyn']

    for method_key in METHODS_TODAY:
        method_display = {
            'smote':      'SMOTE',
            'borderline': 'Borderline-SMOTE',
            'adasyn':     'ADASYN'
        }[method_key]

        print(f"\n{'#'*55}")
        print(f"  BALANCING METHOD: {method_display}")
        print(f"{'#'*55}")

        X_bal, y_bal = apply_balancing(
            X_raw, y_raw, method=method_key)

        for clf_name, pipeline in TODAYS_CLASSIFIERS.items():
            test_one_classifier(
                clf_name, pipeline,
                X_bal, y_bal, method_display)

    print(f"\n\n{'='*55}")
    print(f"  Testing complete!")
    print(f"{'='*55}")
    list_tested_combinations()
