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
		self.name_dict   = {}
		self.custom_tags = False

		os.chdir('../'+D_PATH)


	def plot_options(self, style="whitegrid", use_grid=False, graph_name=True):

		self.style       = style
		self.sns_palette = "Set1"
		self.graph_name  = graph_name


		#sns.set_theme(style=self.style, {'axes.grid' : use_grid})
		sns.set_style(self.style, {'axes.grid' : use_grid})
		#sns.color_palette("Spectral", as_cmap=True)


	def use_tags(self,tags):

		self.custom_tags = True
		self.tag_list    = tags 


	def setup_wahrsager(self):

		self.graph_path      = 'lstm-plots/'
		self.log_path        = 'lstm-logs/'
		#self.compare_df_path = self.graph_path+'standart/'

		self.dont_merge = ['standart','learning_rate','dropout','final']
		
		self.to_merge = [
			'sigmoid','lstm_layers','hidden_layers',
			'past_periods','rolling_mean','rolling_max',
			'max_label_seq','mean_label_seq','test_seq']

		self.all_names = self.dont_merge + self.to_merge

		self.merge_dict = {}
		for name in self.to_merge:
			self.merge_dict[name] = ['standart',name]

		for name in self.dont_merge:
			self.merge_dict[name] = [name]

		self.type_first_split  = 'test_'
		self.type_second_split = '_val-size'

		self.index_name = 'epoch'

		self.name_type_list = ['None']


	def setup_heuristic(self):

		# 'heuristic_'+test_name+'_'+HEURISTIC_TYPE+'_'+str(round(threshold_dem))+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")

		self.graph_path = 'agent-plots/heuristic/'
		self.log_path   = 'agent-logs/'

		self.test_name_list = ['Fine-Tresholds']#['Compare_Approches','Tresholds','Configurations']
		self.heur_name_list = ['Perfekt-Pred']#,'LSTM-Pred','Practical']

		self.all_names = []
		for test_name in self.test_name_list:
			for heur_name in self.heur_name_list:
				self.all_names.append(test_name+'_'+heur_name)

		self.merge_dict = {}
		#self.merge_dict['Compare_Approches'] = []
		for heur_name in self.heur_name_list:
			#self.merge_dict['Compare_Approches'].append('Compare_Approches_'+heur_name)
			self.merge_dict['Fine-Tresholds_'+heur_name] = ['Fine-Tresholds_'+heur_name]
			#self.merge_dict['Configurations_'+heur_name] = ['Configurations_'+heur_name]
				
		self.use_tags(['sum_cost_saving'])

		self.type_first_split  = 'heuristic_'
		self.type_second_split = '_t-stamp'

		self.index_name = 'step'
		'''
		self.name_type_list = []
		for name in self.merge_dict['Compare_Approches']:
			self.name_type_list.append(name)
		'''


	def setup_agents(self):

		self.graph_path = 'agent-plots/heuristic/'
		self.log_path   = 'agent-logs/'

		self.agents_list = ['Q-Table','DQN','DQN+MS','DQN+LSTM']#,'PPO2']

		self.param_dict = {
			'Q-Table': ['standart','learning_rate','gamma','tau',
						'update_num','epsilon_decay','input_list'],

			'DQN': ['standart','learning_rate','gamma','tau',
							'update_num','epsilon_decay','input_list', 'hidden_size'],

			'DQN+MULITSTEP': ['horizon'],

			'DQN+LSTM': ['standart','input_sequence','lstm_size']
			}

		self.all_names = []
		for agent_name in self.param_dict.keys:
			for param_name in self.param_dict[agent_name]:
				self.all_names.append(agent_name+'_'+param_name)

		self.merge_dict = {}
		for agent_name in self.param_dict.keys:
			if self.param_dict[agent_name][0] == 'standart':
				for param_name in self.param_dict[agent_name]:
					self.merge_dict[agent_name+'_'+param_name] = ['standart',param_name]
			else:
				for param_name in self.param_dict[agent_name]:
					self.merge_dict[agent_name+'_'+param_name] = [param_name]

		self.type_first_split  = 'agent_'
		self.type_second_split = '_t-stamp'

		self.index_name = 'step'

		self.name_type_list = ['None']


	def logs_to_csv(self):
		working_dir = os.getcwd()

		for name in self.all_names:

			make_dir(self.graph_path+'seperate_log_csv/'+name)
		
			#try:
			for folder_path in iglob(self.log_path+'*'+name+'*'):
				print('path:',folder_path)
				csv_name = folder_path.split('\\')[-1]
				#name = folder_path.split('/')[-1]
				ea = event_accumulator.EventAccumulator(folder_path)
				ea.Reload()

				#self.csv_path_list = []
				if self.custom_tags == False:
					self.tag_list = []

				for tag in ea.Tags()['scalars']:
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
						self.tag_list.append(tag)
						#self.csv_path_list.append(csv_path)
			
			#except Exception as e:
			#	print('Exception:', e)
			#	print('Could not open any log with path:',self.log_path,'that includes',name)

	
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
						#print(run)
						#print(filename)


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


	def create_graphs(self, plot_type='simple'):
		print(self.tag_list)
		for name in self.merge_dict:
			print(name)
			if plot_type == 'simple':
				self.simple_plot(name)


	def simple_plot(self, name):

		make_dir(self.graph_path+'graphs/'+name)

		for tag in self.tag_list:

			path = self.graph_path+'final_longform_csv/'+name+'/'+name+'_'+tag
			df = pd.read_csv(path+'.csv')
			#df = df.drop(columns=[df.columns[0]])
			print(df)

			
			if self.graph_name == True:
				plt.title(name.replace('_',' '))
			
			sns.lineplot(data=df, dashes=False, x=df.columns[0], y=df.columns[1], hue="type", palette=self.sns_palette)#, label=df.columns[-1])
			
			plt.xlabel(df.index.names[0])
			plt.ylabel(tag)
			plt.savefig(self.graph_path+'graphs/'+name+'/'+name+'_'+tag+'.png')
			plt.close()

			print('Saved graph to:',self.graph_path+'graphs/'+name+'/'+name+'_'+tag+'.png')


	def usual_prep_and_graph_creation(self):
		#self.logs_to_csv()
		self.create_basic_longforms()
		self.merge_longforms()
		self.create_graphs()



def main():

	graph_maker = GraphMaker('_BIG_D/')

	graph_maker.plot_options()
	
	#graph_maker.setup_wahrsager()
	#graph_maker.usual_prep_and_graph_creation()

	graph_maker.setup_heuristic()
	graph_maker.usual_prep_and_graph_creation()

	#graph_maker.setup_heuristic()
	#graph_maker.heuristic_logs_to_csv()
	#graph_maker.create_graphs()

if __name__ == '__main__':
	main()

