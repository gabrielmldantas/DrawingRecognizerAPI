import numpy as np
import pathlib
import time
import statistics
from model_util import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import pickle

CNN_PATH = pathlib.Path('resources/models/cnn.h5')
CSV_DIR_PATH = pathlib.Path('resources/csv/cnn')

def create_cnn_model():
  cnn = Sequential()
  cnn.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
  cnn.add(Conv2D(128, kernel_size=3, activation='relu'))
  cnn.add(Flatten())
  cnn.add(Dense(32, activation='relu'))
  cnn.add(Dense(6, activation='softmax'))
  cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return cnn

def train_cnn():
	data = load_dataset_with_categories()
	X_train, X_test, y_train, y_test = split_data(data['dataset'], data['categories'])

	# 1D -> 2D
	X_train = X_train.reshape([-1, 28, 28, 1])
	X_test = X_test.reshape([-1, 28, 28, 1])

	cnn_param_grid = {'epochs': [3], 'batch_size': [30, 3000]}
	cnn_grid_search = GridSearchCV(KerasClassifier(build_fn=create_cnn_model, class_weight='balanced'), cnn_param_grid, cv=10)
	cnn_grid_result = cnn_grid_search.fit(X_train, y_train)

	create_grid_search_csv(dtree_grid_search)
	create_grid_search_plot()

	metrics = calculate_metrics(cnn_grid_result.best_params_['epochs'], cnn_grid_result.best_params_['batch_size'])
	create_results_csv(metrics)

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

def create_results_csv(results):
    with open(CSV_DIR_PATH.joinpath('samples' + '.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['fit_time', 'predict_time', 'accuracy', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def calculate_metrics(epochs, batch_size):
	results = []
	for i in range(30):
		result = {}
		cnn = create_cnn_model()

		start_time = time.time()
		cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
		end_time = time.time()
		result['fit_time'] = end_time - start_time

		start_time = time.time()
		cnn.predict(X_test)
		end_time = time.time()
		result['predict_time'] = end_time - start_time

		result['f1'] = f1_score(y_test, np.argmax(cnn.predict(X_test), 1), average='macro')
		result['recall'] = recall_score(y_test, np.argmax(cnn.predict(X_test), 1), average='macro')
		result['accuracy'] = accuracy_score(y_test, np.argmax(cnn.predict(X_test), 1))
		result['precision'] = precision_score(y_test, np.argmax(cnn.predict(X_test), 1), average='macro')
		results.append(result)
	return results

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
        if 'epochs' in params_dict:
            param_values.append(str(params_dict['epochs']))
        if 'batch_size' in params_dict:
            param_values.append(str(params_dict['batch_size']))
        normalized_names.append(', '.join(param_values))
    return normalized_names

if __name__ == '__main__':
	CSV_DIR_PATH.mkdir(parents=True, exist_ok=True)
	train_cnn()
	create_samples_plot()
