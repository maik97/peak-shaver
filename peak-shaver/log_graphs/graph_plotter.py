import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import csv
import time

from glob import iglob

#Version 1

def merge_runs(path,tag,column_name_first_split,column_name_second_split,index_name='step'):
	init_df = True
	for filename in iglob(path+'*'+tag+'.csv'):
		if init_df == True:
			df       = pd.read_csv(filename, delimiter=',')
			filename = filename.split(column_name_first_split)[-1]
			filename = filename.split(column_name_second_split)[0]
			df       = df.drop(columns=['Wall time'])
			df       = df.rename(columns={'Value':filename})
			init_df  = False
		else:
			column   = pd.read_csv(filename, delimiter=',')
			filename = filename.split(column_name_first_split)[-1]
			filename = filename.split(column_name_second_split)[0]
			column   = column.rename(columns={'Value':filename})
			column   = column[filename]
			print(column)
			df       = pd.merge(df, column, left_index=True, right_index=True, how='outer')
	df = df.set_index('Step')
	df.columns = df.columns.str.replace('_',' ')
	df.index.names = [index_name]
	print(df)
	return df

def simple_plot(df, path, graph_name=None, ylabel='loss'):

	if graph_name != None:
		plt.title(graph_name)

	sns.set_theme(style="whitegrid")
	ax = sns.lineplot(data=df, dashes=False)
	ax.set(xlabel=df.index.names[0], ylabel=ylabel)
	plt.savefig(path+df.columns[0].split(' ')[0]+'.png')
	plt.close()
	#plt.show()

def wahrsager_graphs():

	df = merge_runs('lstm/overfitting/','loss','test_','_val-size','epoch')
	simple_plot(df,'lstm/overfitting/')

wahrsager_graphs()

def merge_tensorboard_csv(out_file_name='unnamed'):
	init_df = True
	for filename in iglob('*.csv'):
		if init_df == True:
			df       = pd.read_csv(filename, delimiter=',')
			filename = filename.split('.')[0]+': {} €'.format(round(df['Value'].iloc[-1],2))
			df       = df.drop(columns=['Wall time'])
			df       = df.rename(columns={'Value':filename})
			init_df  = False
		else:
			column   = pd.read_csv(filename, delimiter=',')
			filename = filename.split('.')[0]+': {} €'.format(round(column['Value'].iloc[-1],2))
			column   = column.rename(columns={'Value':filename})
			column   = column[filename]
			print(column)
			df       = pd.merge(df, column, left_index=True, right_index=True, how='outer')

	df = df.set_index('Step')
	print(df)
	df.to_csv('finish/'+out_file_name+'.csv')

	sns.set_theme(style="whitegrid")
	ax = sns.lineplot(data=df, dashes=False)
	ax.set(xlabel='Steps', ylabel='Summe Ersparnis in Euro')
	plt.show()

#merge_tensorboard_csv(out_file_name='PPH_SUM_ALL')







