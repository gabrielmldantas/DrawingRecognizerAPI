import pathlib
import multiprocessing
import pickle
import time
import statistics
import csv
from model_util import *
from tqdm import trange
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate

CSV_DIR_PATH = pathlib.Path('resources/csv/traditional')

def fit_model():
    data = load_dataset_with_categories()
    X_train, X_test, y_train, y_test = split_data(data['dataset'], data['categories'])
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    dtree_grid_params = [
      {'criterion': ['gini', 'entropy']},
      {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']},
      {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2']},
    ]

    dtree_grid_search = GridSearchCV(DecisionTreeClassifier(), dtree_grid_params, n_jobs=multiprocessing.cpu_count(), cv=10, refit=False)
    dtree_grid_search.fit(X_train, y_train)

    create_grid_search_csv(dtree_grid_search)
    create_grid_search_plot()

    results = []
    for i in trange(30):
        classifier = DecisionTreeClassifier(**dtree_grid_search.best_params_)
        score_results = cross_validate(classifier, X_train, y_train, scoring=scorers, cv=10, n_jobs=multiprocessing.cpu_count())
        start_time = time.time()
        classifier.predict(X_test)
        delta = time.time() - start_time
        results.append({'predict_time': delta, 'score_results': score_results})

    create_results_csv(results)

def create_results_csv(results):
    normalized_results = []
    for result in results:
        normalized_results.append({
            'fit_time': statistics.mean(result['score_results']['fit_time']),
            'predict_time': result['predict_time'],
            'accuracy': statistics.mean(result['score_results']['test_accuracy']),
            'precision': statistics.mean(result['score_results']['test_precision']),
            'recall': statistics.mean(result['score_results']['test_recall']),
            'f1': statistics.mean(result['score_results']['test_f1'])
        })

    with open(CSV_DIR_PATH.joinpath('samples' + '.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['fit_time', 'predict_time', 'accuracy', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for result in normalized_results:
            writer.writerow(result)


def create_grid_search_csv(dtree_grid_search):
    results = []
    for index, param in enumerate(dtree_grid_search.cv_results_['params']):
        result = {
            'params': param,
            'mean_fit_time': dtree_grid_search.cv_results_['mean_fit_time'][index],
            'std_fit_time': dtree_grid_search.cv_results_['std_fit_time'][index],
            'mean_test_score': dtree_grid_search.cv_results_['mean_test_score'][index],
            'std_test_score': dtree_grid_search.cv_results_['std_test_score'][index]
        }
        results.append(result)

    with open(CSV_DIR_PATH.joinpath('grid_search' + '.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['params', 'mean_fit_time', 'std_fit_time', 'mean_test_score', 'std_test_score'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def create_grid_search_plot():
    data_frame = pandas.read_csv(CSV_DIR_PATH.joinpath('grid_search.csv'))
    data_frame['mean_test_score'] = data_frame['mean_test_score'] * 100
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    colors = range(len(data_frame['params']))
    scatter = plt.scatter('mean_test_score', 'mean_fit_time', data=data_frame, c=colors)

    plt.xlabel('mean_test_score (%)')
    plt.ylabel('mean_fit_time (s)')
    plt.grid(True)

    fig.legend(handles=scatter.legend_elements()[0], labels=normalize_param_names(data_frame['params']),
        loc='upper center', ncol=5)
    plt.savefig(CSV_DIR_PATH.joinpath(f'grid_search.png'))

def create_samples_plot():
    data_frame = pandas.read_csv(CSV_DIR_PATH.joinpath('samples.csv'))
    data_frame['accuracy'] = data_frame['accuracy'] * 100
    data_frame['f1'] = data_frame['f1'] * 100
    data_frame['precision'] = data_frame['precision'] * 100
    data_frame['recall'] = data_frame['recall'] * 100

    plot_info_by_metric = {
        'accuracy': ('Acurácia (AC)', 'acurácia (%)'),
        'f1': ('F1 (F1)', 'f1 (%)'),
        'precision': ('Precisão (PC)', 'precisão (%)'),
        'recall': ('Recall (RC)', 'recall (%)'),
        'fit_time': ('Tempo de treinamento (FT)', 'tempo de treinamento (s)'),
        'predict_time': ('Tempo de classificação (PT)', 'tempo de classificação (s)')
    }

    for metric in data_frame:
        fig, axs = plt.subplots(1, 1, figsize=(9, 9))
        indices = tuple(range(1, len(data_frame[metric]) + 1))
        plt.bar(indices, data_frame[metric])
        plt.xlabel('Sample')
        plt.ylabel(plot_info_by_metric[metric][1])
        plt.suptitle(plot_info_by_metric[metric][0])
        plt.title('Média: {0}'.format(round(statistics.mean(data_frame[metric]), 2)))
        plt.grid(True)
        plt.xticks(indices, indices)
        plt.savefig(CSV_DIR_PATH.joinpath(f'{metric}.png'))

def normalize_param_names(params):
    normalized_names = []
    for param in params:
        params_dict = eval(param)
        param_values = []
        if 'criterion' in params_dict:
            param_values.append(params_dict['criterion'])
        if 'max_features' in params_dict:
            param_values.append(params_dict['max_features'])
        if 'splitter' in params_dict:
            param_values.append(params_dict['splitter'])
        normalized_names.append(', '.join(param_values))
    return normalized_names

if __name__ == '__main__':
    CSV_DIR_PATH.mkdir(parents=True, exist_ok=True)
    fit_model()
    create_samples_plot()
