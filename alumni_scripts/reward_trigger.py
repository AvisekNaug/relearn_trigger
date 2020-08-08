"""
This script can be used to calculate controller reward during demo online deployment. 
The reward structure should be identical to the alumni_env.py file. Effort is being made
in keeping a single reward function for both alumni_env.py and here.
*TODO: will update the description as development is completed
"""


import numpy as np
from pandas import DataFrame

import warnings
with warnings.catch_warnings():
	import tensorflow as tf
	from keras.models import load_model

class deployment_reward_processor():

	def __init__(self, *args, **kwargs):

		# logger
		self.log = kwargs['logger']

		try:
			# the observation and action space variable names
			self.s_name, self.a_name = kwargs['obs_space_vars'], kwargs['action_space_vars']
			# whether data provided to calculate reward is normalized or not
			self.nomralized_data = kwargs['normalized_data']
			# scaler from meta_data_
			self.scaler = kwargs['scaler']
			# path to models
			self.dynamic_model_path = kwargs['model_path']
			# input variables for models
			self.cwe_input_vars, self.cwe_input_shape = kwargs['cwe_input_vars'],kwargs['cwe_input_shape']
			self.hwe_input_vars, self.hwe_input_shape = kwargs['hwe_input_vars'],kwargs['hwe_input_shape']
			self.vlv_input_vars, self.vlv_input_shape = kwargs['vlv_input_vars'],kwargs['vlv_input_shape']
			# test whether state and action variables are part of the scaler variables if data is not normalized
			if not self.nomralized_data:
				assert all(i in self.scaler.columns for i in self.s_name+self.a_name), \
					"Passed non-normalized data with without scaler information for all columns"
			

			self.d_graph = tf.Graph()
			self.d_session = tf.Session(graph=self.d_graph)

			# load all the models
			with self.d_graph.as_default():  # pylint: disable=not-context-manager
				with self.d_session.as_default():  # pylint: disable=not-context-manager
					self.cwe_energy_model = load_model(self.dynamic_model_path+'cwe_best_model')
					self.hwe_energy_model = load_model(self.dynamic_model_path+'hwe_best_model')
					self.vlv_energy_model = load_model(self.dynamic_model_path+'vlv_best_model')

		except Exception as e:
			self.log.error('Reward Trigger Init Module: %s', str(e))
			self.log.debug(e, exc_info=True)

	
	def reload_models(self, *args, **kwargs):

		try:
			with self.d_graph.as_default():  # pylint: disable=not-context-manager
				with self.d_session.as_default():  # pylint: disable=not-context-manager
					self.cwe_energy_model.load_weights(self.dynamic_model_path +'cwe_best_model')
					self.hwe_energy_model.load_weights(self.dynamic_model_path +'hwe_best_model')
					self.vlv_energy_model.load_weights(self.dynamic_model_path +'vlv_best_model')
		
		except Exception as e:
			self.log.error('Reward Trigger Reload Module: %s', str(e))
			self.log.debug(e, exc_info=True)


	def calculate_deployment_reward(self, *args, **kwargs):
		
		try:
			# the obs variables of the environment and action taken by the deployed agent
			s : DataFrame = kwargs['vars_next']
			rl_a = kwargs['rl_set_point']
			rbc_a  = kwargs['rbc_set_point']

			if not self.nomralized_data:
				s[s.columns] = self.scaler.minmax_scale(s, s.columns, s.columns)
				rl_a = self.scaler.minmax_scale(rl_a, self.a_name, self.a_name)
				rbc_a = self.scaler.minmax_scale(rbc_a, self.a_name, self.a_name)
			
			reward_params = {'energy_saved': 100.0, 'energy_savings_thresh': 0.0,
								'energy_penalty': -100.0, 'energy_reward_weight': 0.6,
								'comfort': 10, 'comfort_thresh': 0.10,
								'uncomfortable': 10, 'comfort_reward_weight': 0.2,
								'heating_reward_weight':0.2,
								'action_minmax':[np.array([65]), np.array([72])]
								}
			
			
			'''comfort energy'''
			rl_cwe, rl_hwe = self.calculate_energy(s,rl_a)
			rbc_cwe, rbc_hwe = self.calculate_energy(s,rbc_a)
			rl_energy = rl_cwe + rl_hwe
			rbc_energy = rbc_cwe + rbc_hwe
			reward_energy = reward_params['energy_saved']*(rbc_energy-rl_energy) \
				if rbc_energy-rl_energy>reward_params['energy_savings_thresh'] \
				else reward_params['energy_penalty']*(-rbc_energy+rl_energy)

			'''comfort reward '''
			T_rl_disch = rl_a
			# extract vrf average setpoint temperature
			avg_vrf_stpt = s.loc[s.index[0], 'avg_stpt']
			reward_comfort = reward_params['comfort']/(abs(T_rl_disch-avg_vrf_stpt) + 0.1) \
			if abs(T_rl_disch-avg_vrf_stpt) < reward_params['comfort_thresh'] \
			else reward_params['uncomfortable']*abs(T_rl_disch-avg_vrf_stpt)

			'''reward less heating'''
			oat_t = s.loc[s.index[0], 'oat']
			if (oat_t>0.64):  # warm weather > 68F
				reward_heating = -10.0*T_rl_disch
			else:
				reward_heating = -0.1*T_rl_disch

			reward = reward_params['energy_reward_weight']*reward_energy + \
				reward_params['comfort_reward_weight']*reward_comfort + reward_params['heating_reward_weight']*reward_heating

			return reward
		
		except Exception as e:
			self.log.error('Reward Trigger Calculate Deplyment Reward Module: %s', str(e))
			self.log.debug(e, exc_info=True)

	def calculate_energy(self,s,a):

		try:
			# do not modify original state s
			obs = s.copy()
			# change old actions to new actions
			obs.loc[:, self.a_name] = a
			# create the needed column
			obs['sat-oat'] = obs['sat']-obs['oat']

			"""cololing energy prediction"""
			# get input to cwe model
			cwe_in_obs = obs.loc[:, self.cwe_input_vars]  # in_obs.loc[:, self.cwe_input_vars]
			# convert to array and reshape
			cwe_in_obs = cwe_in_obs.to_numpy().reshape(self.cwe_input_shape)
			# calculate cooling energy
			with self.d_graph.as_default():  # pylint: disable=not-context-manager
				with self.d_session.as_default():  # pylint: disable=not-context-manager
					cooling_energy = float(self.cwe_energy_model.predict(cwe_in_obs, batch_size=1).flatten())

			"""valve state prediction"""
			# now select the inputs
			vlv_in_obs = obs.loc[:, self.vlv_input_vars]
			# convert to array and reshape
			vlv_in_obs = vlv_in_obs.to_numpy().reshape(self.vlv_input_shape)
			# calculate valve state as 0 = off or 1 = on
			with self.d_graph.as_default():  # pylint: disable=not-context-manager
				with self.d_session.as_default():  # pylint: disable=not-context-manager
					valve_state =  np.argmax(self.vlv_energy_model.predict(vlv_in_obs, batch_size=1).flatten())

			"""heating energy prediction only if valve is on = 1"""
			if valve_state == 1:
				# now select the inputs
				hwe_in_obs = obs.loc[:, self.hwe_input_vars]
				# convert to array and reshape
				hwe_in_obs = hwe_in_obs.to_numpy().reshape(self.hwe_input_shape)
				# calculate heating energy
				with self.d_graph.as_default():  # pylint: disable=not-context-manager
					with self.d_session.as_default():  # pylint: disable=not-context-manager
						heating_energy =  float(self.hwe_energy_model.predict(hwe_in_obs, batch_size=1).flatten())
			else:
				heating_energy = 0.0

			return cooling_energy, heating_energy
		
		except Exception as e:
			self.log.error('Reward Trigger Calculate Energy Module: %s', str(e))
			self.log.debug(e, exc_info=True)
