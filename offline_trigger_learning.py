"""
This script implements the different threads that are going to be executed for the Alumni Hall deployment.
This script will control the "Supply Air Set Point" for the Air Handling Units. The skeleton script 
is now being created to formalize the main components needed in creating the relearning agent.
"""
import os
# Enable '0' or disable '' GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from multiprocessing import Event, Lock
from threading import Thread
from datetime import datetime
import json

# ------------From Ibrahim's controller.py script
import sys
import logging
# Set up logging with a global variable "log"
logging.captureWarnings(True)
log = logging.getLogger(__name__)
_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
_handler = logging.StreamHandler(sys.stderr)
# Initially, before the code in __main__ guard executes, logging is set to DEBUG
# to print out all messages to console.
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_formatter)
log.addHandler(_handler)
# ------------From Ibrahim's controller.py script

import warnings
with warnings.catch_warnings():
	from alumni_scripts import model_learn as mdlearn
	from alumni_scripts import control_learn as ctlearn
	from alumni_scripts import alumni_data_utils as a_utils
	from alumni_scripts import deploy_offline as doffline
	from source import utils

if __name__ == "__main__":

	try:
		
		# ------------From Ibrahim's controller.py script
		# Specifing the log file name
		_logfile_handler = logging.FileHandler(filename='log.txt')
		_logfile_handler.setLevel(logging.DEBUG)    # DEBUG is the lowest severity. It means print all messages.
		_logfile_handler.setFormatter(_formatter)   # Set up the format of log messages
		log.addHandler(_logfile_handler)            # add this handler to the logger
		# set up logging severity
		log.setLevel(logging.DEBUG)
		# ------------From Ibrahim's controller.py script
	
		exp_params = {}
		# interval num for relearning : look at logs/Interval{} and write next number to prevent overwrite
		interval = 1
		# how to set prediction sections
		# relearn_interval_kwargs = {'days':0, 'hours':12, 'minutes':0, 'seconds':0}
		# weeks to look back into for retraining
		retrain_range_weeks = 15
		# weeks to train rl on 
		retrain_range_rl_weeks = 5
		# use validation loss in lstm or not
		use_val = True
		# number of epochs to train dynamic models
		epochs = 20000
		# period of data; 1 => 5 mins, 6 => 30 mins
		period = 6 
		# num of steps to learn rl in each train method
		rl_train_steps = int((60/(period*5))*24*7*retrain_range_rl_weeks*40)
		# reinitialize agent at the end of every learning iteration
		reinit_agent = True
		# time stamp of the last time point in the 1 week test data; used to get tsdb data call
		time_stamp = datetime(year = 2019, month = 5, day = 1, hour=0, minute=0, second=0)
		# when to end the experiment
		end_stamp = datetime(year = 2019, month = 12, day = 1, hour=0, minute=0, second=0)
		# time duration to look back at for querying deployment data
		lookback_dur_min = 40
		# week_num to end
		week2end = 52
		# reinitialize agent at the end of every learning iteration
		reinit_agent = False

		save_path = 'tmp/'
		model_path = 'models/'
		log_path = 'logs/'
		results = 'results/'
		trend_data = 'data/trend_data/'
		cwe_data = save_path + 'cwe_data/'
		hwe_data = save_path + 'hwe_data/'
		vlv_data = save_path + 'vlv_data/'
		env_data = save_path + 'env_data/'
		rl_perf_data = save_path + 'rl_perf_data/'
		online_mode = False

		utils.make_dirs(cwe_data)
		utils.make_dirs(hwe_data)
		utils.make_dirs(vlv_data)
		utils.make_dirs(env_data)
		utils.make_dirs(model_path)
		utils.make_dirs(log_path)
		# utils.make_dirs(results)
		utils.make_dirs(rl_perf_data)
		# utils.make_dirs(trend_data)

		# path to saved agent weights
		best_rl_agent_path = 'models/best_rl_agent'

		exp_params['cwe_model_config'] = {
			'model_type': 'regresion', 'train_batchsize' : 32,
			'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
			'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
			'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
			'save_path': cwe_data, 'model_path': model_path, 'name': 'cwe', 'epochs' : epochs
		}
		cwe_vars = ['pchw_flow', 'oah', 'wbt',  'sat', 'oat', 'cwe']

		exp_params['hwe_model_config'] = {
			'model_type': 'regresion', 'train_batchsize' : 32,
			'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
			'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
			'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
			'save_path': hwe_data, 'model_path': model_path, 'name': 'hwe', 'epochs' : epochs
		}
		hwe_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']
		
		exp_params['vlv_model_config'] = {
			'model_type': 'classification', 'train_batchsize' : 32,
			'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
			'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
			'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
			'save_path': vlv_data, 'model_path': model_path, 'name': 'vlv', 'epochs' : epochs
		}
		vlv_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']

		exp_params['env_config'] = {
			'save_path' : env_data, 'model_path': model_path, 'logs' : log_path,
			'obs_space_vars' : ['oat', 'oah', 'wbt', 'avg_stpt', 'sat'], 
			'action_space_vars' :['sat'], 
			'cwe_inputs' : ['sat-oat', 'oah', 'wbt', 'pchw_flow'],
			'hwe_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
			'vlv_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
		}

		# Events
		lstm_data_available = Event()  # new data available for lstm relearning
		end_learning = Event()  # end the relearning procedure
		env_data_available = Event()  # new data available for alumni env rl learning
		lstm_weights_available = Event()  # trained lstm models are avilable
		agent_model_available = Event()  # trained controller weights are availalbe for "online" deployment
		agent_weights_available = Event()  # agent weights are available to be read by deploy loop
		# Locks
		lstm_train_data_lock = Lock()  # read / write lstm data without access issues
		lstm_weights_lock = Lock()  # read / write lstm weights without access issues
		env_train_data_lock = Lock()  # read / write env data without access issues
		agent_weights_lock = Lock()  #  read / write rl weights without access issues
		
		# get agg type and data stats from meta_data.json
		with open('alumni_scripts/meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)
		agg = meta_data_['column_agg_type']
		scaler = a_utils.dataframescaler(meta_data_['column_stats_half_hour'])
		
		reward_processor_params = {
					'logger' : log,
					'obs_space_vars' : exp_params['env_config']['obs_space_vars'],
					'action_space_vars' : exp_params['env_config']['action_space_vars'],
					'normalized_data' : True,
					'scaler' : scaler,
					'model_path' : model_path,
					'lstm_weights_lock':lstm_weights_lock,
					'cwe_input_vars' : exp_params['env_config']['cwe_inputs'],
					'cwe_input_shape' : (-1,1,len(exp_params['env_config']['cwe_inputs'])),
					'hwe_input_vars' : exp_params['env_config']['hwe_inputs'],
					'hwe_input_shape' : (-1,1,len(exp_params['env_config']['hwe_inputs'])),
					'vlv_input_vars' : exp_params['env_config']['vlv_inputs'],
					'vlv_input_shape' : (-1,1,len(exp_params['env_config']['vlv_inputs'])),
				}


		offline_data_gen_params = {'lstm_data_available':lstm_data_available,
									'lstm_train_data_lock':lstm_train_data_lock,
									'lstm_weights_lock':lstm_weights_lock,
									'retrain_range_weeks':retrain_range_weeks,
									'retrain_range_rl_weeks':retrain_range_rl_weeks,
									'env_data_available':env_data_available,
									'env_train_data_lock':env_train_data_lock,
									'agg' : agg,
									'scaler' : scaler,
									'cwe_vars': cwe_vars,
									'hwe_vars': hwe_vars,
									'vlv_vars': vlv_vars,
									'database':'bdx_batch_db',
									'measurement':'alumni_data_v2',
									'save_path': save_path,
									'week2end' : week2end,
									'logger':log}

		deploy_ctrl_th = Thread(target=doffline.deploy_control, daemon = False,
							kwargs={'agent_weights_available' : agent_weights_available,
									'agent_weights_lock' : agent_weights_lock,
									'lstm_weights_lock' : lstm_weights_lock,
									'obs_space_vars' : exp_params['env_config']['obs_space_vars'],
									'scaler' : scaler,
									'best_rl_agent_path' : best_rl_agent_path,
									'period' : period,
									'end_learning': end_learning,
									'logger':log,
									'reward_processor_params':reward_processor_params,
									'offline_data_gen_params':offline_data_gen_params,
									'time_stamp':time_stamp,
									'end_stamp':end_stamp,
									'lookback_dur_min':lookback_dur_min,
									'database':'bdx_batch_db',
									'measurement':'alumni_data_v2'})
		deploy_ctrl_th.start()

		

		model_learn_th = Thread(target=mdlearn.data_driven_model_learn, daemon = False,
							kwargs={'lstm_data_available':lstm_data_available,
									'end_learning':end_learning,
									'lstm_train_data_lock':lstm_train_data_lock,
									'lstm_weights_lock':lstm_weights_lock,
									'lstm_weights_available':lstm_weights_available,
									'cwe_model_config':exp_params['cwe_model_config'],
									'hwe_model_config':exp_params['hwe_model_config'],
									'vlv_model_config':exp_params['vlv_model_config'],
									'save_path': save_path,
									'logger':log,
									'use_val':use_val})
		model_learn_th.start()

		ctrl_learn_th = Thread(target=ctlearn.controller_learn, daemon = False,
							kwargs={
								'env_config':exp_params['env_config'],
								'env_data_available' : env_data_available,
								'lstm_weights_available' : lstm_weights_available,
								'agent_weights_available' : agent_weights_available,
								'end_learning':end_learning,
								'env_train_data_lock' : env_train_data_lock,
								'lstm_weights_lock' : lstm_weights_lock,
								'agent_weights_lock' : agent_weights_lock,
								'rl_train_steps' : rl_train_steps,
								'rl_perf_data' : rl_perf_data,
								'interval' : interval,
								'online_mode' : online_mode,
								'reinit_agent' : reinit_agent,
								'logger':log,})
		ctrl_learn_th.start()

		try:
			log.info('-----------------------All threads started-----------------')
			ctrl_learn_th.join()
			model_learn_th.join()
			deploy_ctrl_th.join()
		except KeyboardInterrupt:
			end_learning.set()
			log.info('Safely exiting')
		log.info("Main Thread : Offine Training Finished")

	except Exception as e:
		log.critical('Could not launch script:\n%s', str(e))
		log.debug(e, exc_info=True)
		exit(-1)