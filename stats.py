import pathlib
import pandas
from scipy.stats import shapiro, wilcoxon

TRADITIONAL_SAMPLES_CSV = pathlib.Path('resources/csv/traditional/samples.csv')
CNN_SAMPLES_CSV = pathlib.Path('resources/csv/cnn/samples.csv')

def calculate_shapiro(samples_file):
	data_frame = pandas.read_csv(samples_file)
	results = {}
	for metric in data_frame:
		results[metric] = shapiro(data_frame[metric])[1]
	
	print(f'Shapiro-Wilk results for {samples_file}')
	normal_metrics = []
	for metric in results:
		p_value = results[metric]
		print(f'{metric}: {p_value}')
		if p_value > 0.05:
			normal_metrics.append(metric)
	print(f'Normal distribution for metrics: {normal_metrics}')
	print()
	return normal_metrics

def calculate_wilcoxon():
	data_frame_traditional = pandas.read_csv(TRADITIONAL_SAMPLES_CSV)
	data_frame_cnn = pandas.read_csv(CNN_SAMPLES_CSV)
	for metric in data_frame_traditional:
		p_value = wilcoxon(data_frame_traditional[metric], data_frame_cnn[metric])
		print(f'{metric}: {p_value}')

if __name__ == '__main__':
	traditional_normal_metrics = calculate_shapiro(TRADITIONAL_SAMPLES_CSV)
	cnn_normal_metrics = calculate_shapiro(CNN_SAMPLES_CSV)
	calculate_wilcoxon()