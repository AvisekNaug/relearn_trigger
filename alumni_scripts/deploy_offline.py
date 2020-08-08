"""
This script containts methods that will observe the current state of the Alumni Hall
and issue a temperature set point to be sent as the set point for the building.
"""
from os import path
import numpy as np
from pandas import read_csv, to_datetime, DataFrame
import json
from datetime import datetime, timedelta
from multiprocessing import Event, Lock
import time
import pytz
from dateutil import tz
from influxdb import DataFrameClient

import warnings
with warnings.catch_warnings():
	from stable_baselines import PPO2
	from alumni_scripts import data_process as dp
	from alumni_scripts import alumni_data_utils as a_utils
	from alumni_scripts import reward_trigger as rt
	from alumni_scripts import data_generator as datagen
	from CoolProp.HumidAirProp import HAPropsSI

def deploy_control(*args, **kwargs):
	"""
	Generates the actual actions to be sent to the actual building
	"""
	# logger
	log = kwargs['logger']
	try:
		with open('alumni_scripts/meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)
		if not path.exists("experience.csv"):
			with open('experience.csv', 'a+') as cfile:
				cfile.write('{},{},{},{},{},{},{},{}\n'.format('time', 'oat', 'oah', 'wbt',
				'avg_stpt', 'sat', 'rlstpt', 'hist_stpt'))
			cfile.close()
		with open('deployment_reward.csv', 'w') as cfile:
			cfile.write('{},{}\n'.format('time','reward'))
		cfile.close()

		agent_weights_available : Event = kwargs['agent_weights_available']  # deploy loop can read the agent weights now
		end_learning : Event = kwargs['end_learning']
		agent_weights_lock : Lock = kwargs['agent_weights_lock']  # prevent data read/write access
		
		# check variables if needed
		obs_space_vars : list = kwargs['obs_space_vars']
		scaler : a_utils.dataframescaler = kwargs['scaler']
		stpt_delta = np.array([0.0]) # in delta F
		stpt_unscaled = np.array([68.0])  # in F
		stpt_scaled = scaler.minmax_scale(stpt_unscaled, ['sat'], ['sat'])
		not_first_loop = False
		period = kwargs['period']
		time_stamp = kwargs['time_stamp']
		lookback_dur_min = kwargs['lookback_dur_min']
		measurement = kwargs['measurement']
		# database client
		client = DataFrameClient(host='localhost', port=8086, database=kwargs['database'],)

		# variables for rewrd trigger module
		reward_processor_params = kwargs['reward_processor_params']
		offline_data_gen_params = kwargs['offline_data_gen_params']

		# initiate reward_trigger class
		rwd_processor = rt.deployment_reward_processor(**reward_processor_params)


		# an initial trained model has to exist
		log.info('Deploy Control Module: Controller not ready for deployment. Wait for some time')
		agent_weights_available.wait()
		log.info('Deploy Control Module: Controller is ready for initial deployment')
		with agent_weights_lock:
			rl_agent = PPO2.load(kwargs['best_rl_agent_path'])
		agent_weights_available.clear()
		log.info('Deploy Control Module: Controller Weights Read from offline phase')

		while not end_learning.is_set():
		
			# get current scaled and uncsaled observation
			df, df_unscaled, hist_stpt, hist_stpt_scaled = get_real_obs(client, time_stamp, meta_data_, obs_space_vars,
														 scaler, period, log,
														lookback_dur_min, measurement)
			curr_obs_scaled = df.to_numpy().flatten()
			curr_obs_unscaled = df_unscaled.to_numpy().flatten()

			# if we want to set the sat to the exact value from previous time step
			# comment it out if not
			if not_first_loop:
				curr_obs_scaled[-1] = stpt_scaled[0]
				curr_obs_unscaled[-1] = stpt_unscaled[0]
			else:
				not_first_loop = True

			# check individual values to lie in appropriate range
			# already done by online_data_clean method

			# check individual values to not move too much from previous value
			# nominal values already checked within online_data_clean method

			# calculate the reward value
			rwd = rwd_processor.calculate_deployment_reward(**{'vars_next' : df,
															   'rl_env_action' : stpt_scaled,
															   'rbc_env_action' : hist_stpt_scaled})
			# save rwd into a csv file
			with open('deployment_reward.csv', 'a+') as cfile:
					cfile.write('{},{:.3f}\n'.format( time_stamp, float(rwd) ))
			cfile.close()

			# TODO: decide whether relearning has to happen or not
			relearn_triggered = False
			if relearn_triggered:
				log.info('Deploy Control Module: Relearn has been Triggered')
				# create train data
				offline_data_gen_params.update({'time_stamp': time_stamp})
				datagen.offline_data_gen(**offline_data_gen_params)
				log.info('Deploy Control Module: Wait until controller has been relearned')
				agent_weights_available.wait() # wait until new controller weights are generated
				log.info('Deploy Control Module: Controller has been relearned')

									   

			# get new agent model in case it is available
			if agent_weights_available.is_set():
				with agent_weights_lock:
					rl_agent.load_parameters(kwargs['best_rl_agent_path'])
				agent_weights_available.clear()
				log.info('Deploy Control Module: Controller Weights Adapted')

			# predict new delta and add it to new temp var for next loop check
			stpt_delta = rl_agent.predict(curr_obs_scaled)
			log.info('Deploy Control Module: Current SetPoint: {}'.format(curr_obs_unscaled[-1]))
			log.info('Deploy Control Module: Suggested Delta: {}'.format(stpt_delta[0][0]))
			stpt_unscaled[0] = curr_obs_unscaled[-1] + float(stpt_delta[0])  # stpt_unscaled[0] 
			# clip it in case it crosses a range
			stpt_unscaled = np.clip(stpt_unscaled, np.array([65.0]), np.array([72.0]))
			# scale it
			stpt_scaled = scaler.minmax_scale(stpt_unscaled, ['sat_stpt'], ['sat_stpt'])

			# write it to a file for BdX
			with open('reheat_preheat_setpoint.txt', 'w') as cfile:
				cfile.seek(0)
				cfile.write('{}\n'.format(str(stpt_unscaled[0])))
			cfile.close()

			# write output to file for our use
			fout = np.concatenate((curr_obs_unscaled, stpt_unscaled, hist_stpt))
			with open('experience.csv', 'a+') as cfile:
				cfile.write('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(datetime.now(), fout[0],
				fout[1], fout[2], fout[3], fout[4], fout[5], fout[6]))
			cfile.close()

			# advance by 30 mins for next step
			time_stamp += timedelta(**{'days':0, 'hours':0, 'minutes':30, 'seconds':0})

			# TODO: decide how to set end learning

	except Exception as e:
		log.error('Deploy Control Module: %s', str(e))
		log.debug(e, exc_info=True)



def get_real_obs(client, time_stamp, meta_data_: dict, obs_space_vars : list, scaler, period, log, 
							lookback_dur_min, measurement):

	try:

		log.info('Deploy Control Module: Getting Data from TSDB')
		result_obj = client.query("select * from {} where time >= '{}' - {}w \
								and time < '{}'".format(measurement, \
								str(time_stamp),lookback_dur_min,str(time_stamp)))
		df_= result_obj[measurement]
		df_ = df_.drop(columns = ['data_cleaned', 'aggregated', 'time-interval'])

		# collect current set point
		hist_stpt = df_.loc[df_.index[-1],['sat_stpt']].to_numpy().copy().flatten()

		# clip less than 0 values
		df_.clip(lower=0, inplace=True)

		# aggregate data
		rolling_sum_target, rolling_mean_target = [], []
		for col_name in df_.columns:
			if meta_data_['column_agg_type'][col_name] == 'sum' : rolling_sum_target.append(col_name)
			else: rolling_mean_target.append(col_name)
		df_[rolling_sum_target] =  a_utils.window_sum(df_, window_size=6, column_names=rolling_sum_target)
		df_[rolling_mean_target] =  a_utils.window_mean(df_, window_size=6, column_names=rolling_mean_target)
		df_ = a_utils.dropNaNrows(df_)

		# Sample the last half hour data
		df_ = df_.iloc[[-1],:]

		# also need an unscaled version of the observation for logging
		df_unscaled = df_.copy()

		# scale the columns: here we will use min-max
		df_[df_.columns] = scaler.minmax_scale(df_, df_.columns, df_.columns)

		hist_stpt_scaled = df_.loc[df_.index[-1],['sat_stpt']].to_numpy().copy().flatten()

		# create avg_stpt column
		stpt_cols = [ele for ele in df_.columns if 'vrf' in ele]
		df_['avg_stpt'] = df_[stpt_cols].mean(axis=1)
		# drop individual set point cols
		df_.drop( columns = stpt_cols, inplace = True)
		# rearrange observation cols
		df_ = df_[obs_space_vars]

		# create avg_stpt column
		stpt_cols = [ele for ele in df_unscaled.columns if 'vrf' in ele]
		df_unscaled['avg_stpt'] = df_unscaled[stpt_cols].mean(axis=1)
		# drop individual set point cols
		df_unscaled.drop( columns = stpt_cols, inplace = True)
		# rearrange observation cols
		df_unscaled = df_unscaled[obs_space_vars] 

		return df_, df_unscaled, hist_stpt, hist_stpt_scaled

	except Exception as e:
		log.error('Deploy Control Module: %s', str(e))
		log.debug(e, exc_info=True)




