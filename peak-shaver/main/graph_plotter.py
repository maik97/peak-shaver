import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import csv
import time

from glob import iglob, glob

try:
    from tensorflow.python.summary import event_accumulator
except:
    from tensorboard.backend.event_processing import event_accumulator

try:
	from main.common_func import make_dir
except:
	from common_func import make_dir

def load_tensorboard_logs(D_PATH, log_from, partial_name):
	
	graph_path = '../'+D_PATH+log_from+'-plots/'+partial_name
	log_path = '../'+D_PATH+log_from+'-logs/'
	make_dir(graph_path)

	try:
		for folder_path in iglob(log_path+'*'+partial_name+'*'):
			print('path:',folder_path)
			name = folder_path.split('\\')[-1]
			#name = folder_path.split('/')[-1]
			ea = event_accumulator.EventAccumulator(folder_path)
			ea.Reload()

			for tag in ea.Tags()['scalars']:
				pd.DataFrame(ea.Scalars(tag)).to_csv(graph_path+'/'+name+'-tag-'+tag+'.csv')
	
	except Exception as e:
		print('Exception:', e)
		print('Could not open any log with path:',graph_path)


def merge_runs(path,tag,column_name_first_split,column_name_second_split,index_name='step'):
	
	init_df = True
	for filename in iglob(path+'*-tag-'+tag+'.csv'):
		if init_df == False:
			column   = pd.read_csv(filename, delimiter=',')
			filename = filename.split(column_name_first_split)[-1]
			filename = filename.split(column_name_second_split)[0]
			column   = column.rename(columns={'Value':filename})
			column   = column.rename(columns={'value':filename})
			column   = column[filename]
			df       = pd.merge(df, column, left_index=True, right_index=True, how='outer')			

		else:
			df       = pd.read_csv(filename, delimiter=',', index_col=0)
			filename = filename.split(column_name_first_split)[-1]
			filename = filename.split(column_name_second_split)[0]
			try:
				df   = df.drop(columns=['Wall time'])
			except:
				df   = df.drop(columns=['wall_time'])
			df       = df.rename(columns={'Value':filename})
			df       = df.rename(columns={'value':filename})
			init_df  = False
		print(df)
	
	#try:		
	#	df = df.set_index('Step')
	#except:
	df = df.set_index('step')
	df.columns = df.columns.str.replace('_',' ')
	df.index.names = [index_name]
	return df


def simple_plot(df, path, tag, ylabel='', graph_name=None):

	if graph_name != None:
		plt.title(graph_name)

	sns.set_theme(style="whitegrid")
	#sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
	#plt.rcParams["font.weight"] = "bold"
	#plt.rcParams["axes.labelweight"] = "bold"
	ax = sns.lineplot(data=df, dashes=False, sort=False)
	ax.set(xlabel=df.index.names[0], ylabel=ylabel)
	plt.savefig(path+df.columns[-1].split(' ')[0]+'_'+tag+'.png')
	plt.close()
	print('Saved graph to:',path+df.columns[-1].split(' ')[0]+'_'+tag+'.png')
	#plt.show()


def wahrsager_plot(path,tag,standart_df=None):
	
	try:
		# Get merged dataframe of the runs:
		df = merge_runs(path,tag,'test_','_val-size','epoch')
		
		# Add standart run to compare:
		if isinstance(standart_df, pd.DataFrame):
			column   = standart_df.rename(columns={'value':'standart'})
			column   = column['standart']
			df       = pd.merge(df, column, left_index=True, right_index=True, how='outer')	
			
			# Rearrange column order:
			cols = df.columns.tolist()
			cols = cols[-1:] + cols[:-1]
			df = df[cols]
			print(df)

		# Create and save graph:
		simple_plot(df, path, tag, ylabel=tag)
	
	except Exception as e:
		print('Exception:', e)
		print('Could not create graph for',tag,'with path:',path)


def wahrsager_graphs(D_PATH='_BIG_D/'):

	partial_name_list = ['standart','learning_rate','dropout']
	
	partial_name_list_standart_compare = [
		'sigmoid','lstm_layers','hidden_layers','past_periods',
		'rolling_mean','rolling_max','max_label_seq','mean_label_seq','test_seq']

	tag_list = ['loss','mae','val_loss','val_mae']

	
	for partial_name in partial_name_list:
		load_tensorboard_logs(D_PATH, log_from='lstm', partial_name=partial_name)
	for partial_name in partial_name_list_standart_compare:
		load_tensorboard_logs(D_PATH, log_from='lstm', partial_name=partial_name)

	
	for partial_name in partial_name_list:
		graph_path = '../'+D_PATH+'lstm-plots/'+partial_name
		for tag in tag_list:
			wahrsager_plot(graph_path+'/',tag)

	for partial_name in partial_name_list_standart_compare:
		graph_path = '../'+D_PATH+'lstm-plots/'+partial_name
		for tag in tag_list:
			standart_df = pd.read_csv(glob('../'+D_PATH+'lstm-plots/standart/*-tag-'+tag+'.csv')[0])
			wahrsager_plot(graph_path+'/', tag, standart_df=standart_df)



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







