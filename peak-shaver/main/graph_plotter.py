import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import csv
import time

from glob import iglob, glob
from datetime import date, datetime

try:
    from tensorflow.python.summary import event_accumulator
except:
    from tensorboard.backend.event_processing import event_accumulator

try:
	from main.common_func import make_dir
except:
	from common_func import make_dir

class GraphMaker:
	'''
	Creates graphs from the tensorbpoard logs. Make sure to use all test names as they are pre-defined in the code, so the graph maker can read the files correctly.
	'''

	def __init__(self, D_PATH):

		self.D_PATH      = D_PATH
		self.custom_tags = False

		os.chdir('../'+D_PATH)


	def plot_options(self, style="whitegrid", use_grid=False, graph_name=True):

		self.style       = style
		self.sns_palette = "deep"
		self.graph_name  = graph_name
		sns.set_style(self.style, {'axes.grid' : use_grid})


	def use_tags(self,tags):

		self.custom_tags = True
		self.tag_list    = tags 


	def setup_wahrsager(self):

		self.rolling_mean = 10

		self.graph_path      = 'lstm-plots/'
		self.log_path        = 'lstm-logs/'

		self.dont_merge = ['lstm_512_hidden_layers']#'standard','learning_rate','lstm_512_hidden_layers']# ,'dropout','final']

		self.to_merge = []#'activation', 'dropout',
			#'lstm_layers','hidden_layers',
			#'past_periods','mean','max',
			#'max_label_seq','mean_label_seq','seq'
			#]

		self.all_names = self.dont_merge + self.to_merge

		self.merge_dict = {}
		for name in self.to_merge:
			self.merge_dict[name] = ['standard',name]

		for name in self.dont_merge:
			self.merge_dict[name] = [name]

		self.type_first_split  = 'test_'
		self.type_second_split = '_val-size'

		self.index_name = 'epoch'

		self.name_type_list = ['None']


	def setup_heuristic(self):

		self.rolling_mean = 1

		self.graph_path = 'agent-plots/heuristic/'
		self.log_path   = 'agent-logs/'

		self.test_name_list = ['Compare_Approches','Tresholds','Configurations']
		self.heur_name_list = ['Perfekt-Pred','LSTM-Pred','Practical']

		self.all_names = []
		for test_name in self.test_name_list:
			for heur_name in self.heur_name_list:
				self.all_names.append(test_name+'_'+heur_name)

		self.merge_dict = {}
		self.merge_dict['Compare_Approches'] = []
		for heur_name in self.heur_name_list:
			self.merge_dict['Compare_Approches'].append('Compare_Approches_'+heur_name)
			self.merge_dict['Tresholds_'+heur_name] = ['Tresholds_'+heur_name]
			self.merge_dict['Configurations_'+heur_name] = ['Configurations_'+heur_name]
		

		lstm_name_list = ['NORMAL','MAX','MEAN','MAX-LABEL-SEQ','MEAN-LABEL-SEQ','SEQ-MAX','SEQ-MEAN']
		lstm_test_list = ['LSTM-Pred-6-6','LSTM-Pred-12-12','LSTM-Pred-24-24','LSTM-Pred-24-6','LSTM-Pred-24-12']
		for lstm_test in lstm_test_list:
			self.merge_dict[lstm_test] = []
			for lstm_name in lstm_name_list:
				self.merge_dict[lstm_test].append(lstm_test+'_'+lstm_name)
				self.all_names.append(lstm_test+'_'+lstm_nam)

		self.use_tags(['sum_cost_saving'])

		self.type_first_split  = 'heuristic_'
		self.type_second_split = '_t-stamp'

		self.index_name = 'step'

		self.name_type_list = []
		for name in self.merge_dict['Compare_Approches']:
			self.name_type_list.append(name)

		for lstm_test in lstm_test_list:
			for name in self.merge_dict[lstm_test]:
				self.name_type_list.extend(name)


	def setup_agents(self):

		self.rolling_mean = 10

		self.graph_path = 'agent-plots/agents/'
		self.log_path   = 'agent-logs/'

		self.agents_list = ['Compare_Agents','Q-Table','DQN','DQN+MS','DQN+LSTM']#,'PPO2']

		self.param_dict = {

			'Compare_Agents': [#'Q-Table',
								#'DQN',
								'PPO2',
								#'DQN+LSTM'
								]
			
			#'Q-Table': ['standard','learning_rate','gamma','tau',
			#			'update_num','epsilon_decay','input_list'],

			#'DQN': ['standard','learning_rate','gamma','tau',
			#				'update_num','epsilon_decay','input_list', 'hidden_size'],
			
			#'DQN+MS': ['horizon'],
			
			#'DQN+LSTM': ['standard','input_sequence','lstm_size']
			#'Q-Table': ['lstm_inputs'],

			}

		self.all_names = []
		for agent_name in self.param_dict:
			for param_name in self.param_dict[agent_name]:
				self.all_names.append(agent_name+'_'+param_name)

		self.merge_dict = {}
		for agent_name in self.param_dict:
			if self.param_dict[agent_name][0] == 'standard':
				for param_name in self.param_dict[agent_name]:
					self.merge_dict[agent_name+'_'+param_name] = [agent_name+'_standard',agent_name+'_'+param_name]
			else:
				for param_name in self.param_dict[agent_name]:
					self.merge_dict[agent_name+'_'+param_name] = [agent_name+'_'+param_name]

		self.merge_dict['Compare_Agents'] =  ['Compare_Agents_Q-Table',
												'Compare_Agents_DQN',
												'Compare_Agents_PPO2',
												'Compare_Agents_DQN+LSTM'
												]

		self.type_first_split  = 'agent_'
		self.type_second_split = '_t-stamp'

		self.index_name = 'epoch'

		self.name_type_list = []
		for name in self.merge_dict['Compare_Agents']:
			self.name_type_list.append(name)

		self.custom_tags = True
		self.tag_list = [#'discrete-action',
			#'real-GEC',
			#'target-GEC',
			#'SoC-SMS',
			#'SoC-LION',
			#'max-peak-day',
			#'Loss',
			#'Epsilon',
			#'max-peak-week',
			#'max-peak-epoch',
			'sum_savings_epoch',
			#'sum_reward_epoch'
			]


	def logs_to_csv(self):
		working_dir = os.getcwd()

		for name in self.all_names:

			make_dir(self.graph_path+'seperate_log_csv/'+name)
		
			try:
				for folder_path in iglob(self.log_path+'*'+name+'*'):
					print('path:',folder_path)
					csv_name = folder_path.split('\\')[-1]
					#name = folder_path.split('/')[-1]
					ea = event_accumulator.EventAccumulator(folder_path)
					ea.Reload()

					#self.csv_path_list = []
					if self.custom_tags == False:
						self.tag_list = []

					#for tag in self.tag_list:
					for tag in ea.Tags()['scalars']:
						try:
							tag_str = tag.replace(':','').split(' ')[-1]
							print(tag_str)
							#os.chdir(self.graph_path+'seperate_log_csv/'+name+'/')
							csv_path = self.graph_path+'seperate_log_csv/'+name+'/'+csv_name+'-tag-'+tag_str+'.csv'

							#print(pd.DataFrame(ea.Scalars(tag)))
							pd.DataFrame(ea.Scalars(tag)).to_csv(csv_path)
							#print(test)
							#test.to_csv(csv_name+'-tag-'+tag_str+'.csv')
							#os.chdir(working_dir)
							
							if self.custom_tags == False:
								self.tag_list.append(tag_str)
								#self.csv_path_list.append(csv_path)

						except Exception as e:
							print('Exception:', e)
			except Exception as e:
				print('Exception:', e)
				print('Could not open any log with path:',self.log_path,'that includes',name)


	def plot_lstm_output(self):

		path = self.graph_path+'output_graphs/all_runs'
		make_dir(path)


		index_list = []
		max_list   = []
		mean_list  = []
		for csv in iglob('lstm-outputs/*.csv'):
			print(csv)
			csv_name = csv.split('\\')[-1]

			df = pd.read_csv(csv, delimiter=',', index_col=[0])
			df = df.rename(columns={df.columns[0]:'prediction'})	
			df = df.rename(columns={df.columns[1]:'real value'})	
			df = df.rename(columns={df.columns[2]:'absoulte error'})	
			df = df.rename(columns={df.columns[3]:'squared error'})	

			print(df)

			index_list.append(csv_name)
			max_list.append(np.max(df['absoulte error'].to_numpy()))
			mean_list.append(np.mean(df['absoulte error'].to_numpy()))

			df.plot()
			plt.ylabel('energy demand in kw')
			plt.xlabel('step')
			plt.savefig(path+'/'+csv_name+'.png')
			plt.close()

		error_df = pd.DataFrame({
			'run':        index_list,
			'max_error':  max_list,
			'mean_error': mean_list,
			})
		print(error_df)
		error_df.to_csv(self.graph_path+'output_graphs/lstm_error.csv')

	def create_basic_longforms(self):

		for name in self.all_names:

			make_dir(self.graph_path+'basic_longform_csv/'+name)
			make_dir(self.graph_path+'learn_time/'+name)

			learn_time_dict_list = [] 
			for tag in self.tag_list:

				try:
					print('Creating longform for',name,'with',tag,'...')
					df_list = []
					name_list = []
					for filename in iglob(self.graph_path+'seperate_log_csv/'+name+'/*-tag-'+tag+'.csv'):

						# Name:
						df = pd.read_csv(filename, delimiter=',')
						df = df.set_index('step')
						df = df.rename(columns={df.columns[-1]:name})	
						df = df.rename(columns={df.columns[0]:self.index_name})	

						if self.rolling_mean != 1:
							df[name] = df[name].rolling(10).mean()


						# Index:
						df.columns     = df.columns.str.replace('_',' ')
						df.index.names = [self.index_name]

						# drop unnamed:
						df = df.drop(columns=[df.columns[0]])

						# Add column for type:
						if any(type_string in name for type_string in self.name_type_list):
							type_string = name.split('_')[-1]
							df['type']  = type_string
						else:
							type_string = filename.split(self.type_first_split)[-1]
							type_string = type_string.split(self.type_second_split)[0]
							type_string = type_string.split('_')[-1]
							df['type']  = type_string

						# Add column for run number:
						run = name_list.count(type_string)
						df['run']  = run

						# Calculate learning time:
						learn_time_dict_list.append(self.calc_learn_time(df[df.columns[-4]], tag, name, run))
						df = df.drop(columns=[df.columns[-4]])
						
						#df.name = filename
						df_list.append(df)
						name_list.append(type_string)

					# Save longform table:
					concat_df = pd.concat(df_list)
					concat_df.to_csv(self.graph_path+'basic_longform_csv/'+name+'/'+name+'_'+tag+'.csv')

				except Exception as e:
					print(e)
			
			# Save learning time table:
			learn_time_df = pd.DataFrame(learn_time_dict_list)
			learn_time_df.to_csv(self.graph_path+'learn_time/'+name+'/'+name+'_time_comparison.csv')
			'''
			if compare == True:
				learn_time_compare = pd.read_csv(glob(self.compare_df_path+'*_time_comparison.csv')[0], delimiter=',')
				learn_time_compare = learn_time_compare.drop(columns=[learn_time_compare.columns[0]])
				learn_time_df =  pd.concat([learn_time_compare,learn_time_df])
			'''


	def calc_learn_time(self, wall_time_col, tag, name, run):
		
		time_start = datetime.fromtimestamp(wall_time_col.iloc[0])
		time_end   = datetime.fromtimestamp(wall_time_col.iloc[-1])
		learn_time = time_end - time_start

		learn_time_dict = { 
			'name' : name,
			'tag' : tag,
			'run' : run,
			'time' : learn_time,
			}

		return learn_time_dict


	def merge_longforms(self):

		for key in self.merge_dict:

			make_dir(self.graph_path+'final_longform_csv/'+key)

			try:
				for tag in self.tag_list:
					df_list = []
					for name in self.merge_dict[key]:
						try:
							path = self.graph_path+'basic_longform_csv/'+name+'/'+name+'_'+tag+'.csv'
							print('Merge to',key,'from',path)
							df = pd.read_csv(path, delimiter=',', index_col=self.index_name)
							df = df.rename(columns={df.columns[-3]:key})
							df.columns = df.columns.str.replace('_',' ')
							df_list.append(df)
							print(df)
						except Exception as e:
							print(e)

					concat_df = pd.concat(df_list)
					concat_df.to_csv(self.graph_path+'final_longform_csv/'+key+'/'+key+'_'+tag+'.csv')
			except Exception as e:
					print(e)

	def create_graphs(self, plot_type='simple'):
		print(self.tag_list)
		for name in self.merge_dict:
			print(name)
			if plot_type == 'simple':
				self.simple_plot(name)


	def simple_plot(self, name):

		make_dir(self.graph_path+'graphs/'+name)

		for tag in self.tag_list:

			try:
				path = self.graph_path+'final_longform_csv/'+name+'/'+name+'_'+tag
				df = pd.read_csv(path+'.csv')
				#df = df.drop(columns=[df.columns[0]])
				print(df)

				'''
				if self.graph_name == True:
					plt.title(name.replace('_',' '))
				
				sns.lineplot(data=df, dashes=False, x=df.columns[0], y=df.columns[1], hue="type", palette=self.sns_palette)#, label=df.columns[-1])
				
				#plt.xlabel(df.index.names[0])
				plt.xlabel('epoch')
				plt.ylabel(tag)

				#plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
				plt.legend(bbox_to_anchor=(1.02, 1),loc='right')

				#plt.savefig(self.graph_path+'graphs/'+name+'/'+name+'_'+tag+'.png')
				plt.close()
				'''
				fig, ax1 = plt.subplots(1,1)

				g = sns.lineplot(data=df, dashes=False, x=df.columns[0], y=df.columns[1], hue="type", palette=self.sns_palette, ax=ax1)#, label=df.columns[-1])

				'''
				box = g.get_position()
				g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

				# Put a legend to the right side
				g.legend(loc='center right', bbox_to_anchor=(1.33, 0.5), ncol=1)
				'''
				if self.graph_name == True:
					plt.title(name.replace('_',' '))

				plt.ylabel(tag)
				plt.xlabel(self.index_name)
				
				#plt.show()
				plt.savefig(self.graph_path+'graphs/'+name+'/'+name+'_'+tag+'.png')
				plt.close()
				
				print('Saved graph to:',self.graph_path+'graphs/'+name+'/'+name+'_'+tag+'.png')
			
			except Exception as e:
				print(e)

	def usual_prep_and_graph_creation(self):
		self.logs_to_csv()
		self.create_basic_longforms()
		self.merge_longforms()
		self.create_graphs()





def main():

	graph_maker = GraphMaker('_BIG_D/')

	graph_maker.plot_options()
	
	
	graph_maker.setup_wahrsager()
	#graph_maker.plot_lstm_output()

	#graph_maker.setup_heuristic()

	#graph_maker.setup_agents()

	graph_maker.usual_prep_and_graph_creation()

if __name__ == '__main__':
	main()

