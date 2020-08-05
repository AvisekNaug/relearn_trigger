"""
This script will have methods/Threads that help create data for different other threads
1. offline_data_gen:
	Whenever lstm_train_data_lock, env_train_data_lock is free, do in a never-ending loop
	a. read the stored file and create lstm related data for the last 3 months + 1 week
	b. read the stored file and create environment related data for last 3 months + 1 week over same period
	* This method needs to know a timestamp as an input based on which it will get data from offline database.
2. online_data_process : Use stats generated by data_stats to process data online
"""

# imports
import sys, os
import time
import json
import numpy as np
from pandas import read_csv, to_datetime
from datetime import datetime, timedelta
import pytz
from dateutil import tz
from threading import Thread
from multiprocessing import Event, Lock
from influxdb import DataFrameClient
from keras.utils import to_categorical
from alumni_scripts import alumni_data_utils as a_utils
from alumni_scripts import data_process as dp
from CoolProp.HumidAirProp import HAPropsSI
import logging
sys.path.append(os.path.abspath(os.path.join('..')))
import time
import multiprocessing
import psutil


def offline_data_gen(*args, **kwargs):

	# logger
	log = kwargs['logger']
	try:
		time_stamp = kwargs['time_stamp']
		# year and week
		year_num, week_num, _ = time_stamp.isocalendar()

		# Events
		lstm_data_available : Event = kwargs['lstm_data_available']  # new data available for lstm relearning
		env_data_available : Event = kwargs['env_data_available']  # new data available for env relearning  # pylint: disable=unused-variable
		end_learning : Event = kwargs['end_learning'] 

		# Locks
		lstm_train_data_lock : Lock = kwargs['lstm_train_data_lock']  # prevent dataloop from writing data
		env_train_data_lock : Lock = kwargs['env_train_data_lock']  # prevent dataloop from writing data  # pylint: disable=unused-variable

		# relearn interval in date time format
		relearn_interval_kwargs = kwargs['relearn_interval_kwargs']
		# retrain range in weeks
		retrain_range_weeks = kwargs['retrain_range_weeks']
		# week_num to end
		week2end = kwargs['week2end']
 

		client = DataFrameClient(host='localhost', port=8086, database=kwargs['database'],)


		while not end_learning.is_set():

			"""relearning interval decider: here it is a condition; for online it will be time interval or error measure 
			along side the already exisitng conditions"""
			if not (lstm_data_available.is_set() | env_data_available.is_set()):  # or condition prevents faster overwrite for env data

				log.info('OfflineDataGen: Getting Data from TSDB')

				result_obj = client.query("select * from {} where time >= '{}' - {}w \
									and time < '{}'".format(kwargs['measurement'], str(time_stamp),retrain_range_weeks,str(time_stamp)))
				df_env = result_obj[kwargs['measurement']]
				df_env = df_env.drop(columns = ['data_cleaned', 'aggregated', 'time-interval'])

				data_gen_process_cwe_th = Thread(target=data_gen_process_cwe, daemon=False,
											kwargs={ 
											'df' : result_obj[kwargs['measurement']].loc[:,kwargs['cwe_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_hwe_th = Thread(target=data_gen_process_hwe, daemon=False, 
											kwargs={ 
											'df' : result_obj[kwargs['measurement']].loc[:,kwargs['hwe_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_vlv_th = Thread(target=data_gen_process_vlv, daemon=False, 
											kwargs={ 
											'df' : result_obj[kwargs['measurement']].loc[:,kwargs['vlv_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_env_th = Thread(target=data_gen_process_env, daemon=False, 
											kwargs={
											'df' : df_env,
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'],
											'save_path':kwargs['save_path']
											})


				with lstm_train_data_lock:
					data_gen_process_cwe_th.start()
					data_gen_process_hwe_th.start()
					data_gen_process_vlv_th.start()
					data_gen_process_cwe_th.join()
					data_gen_process_vlv_th.join()
					data_gen_process_hwe_th.join()
				lstm_data_available.set()  # data is now available for lstm training
			
				with env_train_data_lock:	
					data_gen_process_env_th.start()
					data_gen_process_env_th.join()
				env_data_available.set()  # data is now available for agent and env training

				log.info('OfflineDataGen: Dynamic Model and Gym Env data available')

				time_stamp += timedelta(**relearn_interval_kwargs)
				week_num += 1
				week_num = week_num if week_num%53 != 0 else 1
				year_num = year_num if week_num!= 1 else year_num+1

				if week_num == week2end:  # can be other terminating condition like year==2020 & week=5 etc
					end_learning.set()
					break
	
	except Exception as e:
		log.error('Off-Line Data Generator Module: %s', str(e))
		log.debug(e, exc_info=True)


def data_gen_process_cwe(*args, **kwargs):

	# read the data from the database
	df = kwargs['df'].copy()

	# smooth the data
	#df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0, inplace=True) # Remove <0 values for all columns as a result of smoothing

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_mean(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)

	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)

	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)

	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']

	# select non-zero operating regions
	df = a_utils.df2operating_regions(df, ['cwe', 'pchw_flow'], [0.001, 0.001])

	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(weeks=6)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_test, X_train, y_test, y_train = a_utils.df_2_arrays(df = df,
		predictorcols = ['sat-oat', 'oah', 'wbt', 'pchw_flow'], outputcols = ['cwe'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)

	# save test ids for later plots
	# idx_end = -max(X_test.shape[1],y_test.shape[1])
	# idx_start = idx_end - X_test.shape[0] + 1
	# test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	# test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	# with open(kwargs['save_path']+'cwe_data/cwe_test_info.txt', 'a') as ifile:
	# 	ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'cwe_data/cwe_X_train.npy', X_train)
	np.save(kwargs['save_path']+'cwe_data/cwe_X_val.npy', X_test)
	np.save(kwargs['save_path']+'cwe_data/cwe_y_train.npy', y_train)
	np.save(kwargs['save_path']+'cwe_data/cwe_y_val.npy', y_test)


def data_gen_process_hwe(*args, **kwargs):
	
	# read the data from the database
	df = kwargs['df'].copy()


	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0, inplace=True) # Remove <0 values for all columns as a result of smoothing

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_mean(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)

	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)

	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)

	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']

	# select non-zero operating regions
	df = a_utils.df2operating_regions(df, ['hwe'], [0.001])

	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(weeks=13)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_test, X_train, y_test, y_train = a_utils.df_2_arrays(df = df,
		predictorcols = ['oat', 'oah', 'wbt', 'sat-oat'], outputcols = ['hwe'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)


	# save test ids for later plots
	# idx_end = -max(X_test.shape[1],y_test.shape[1])
	# idx_start = idx_end - X_test.shape[0] + 1
	# test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	# test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	# with open(kwargs['save_path']+'hwe_data/hwe_test_info.txt', 'a') as ifile:
	# 	ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'hwe_data/hwe_X_train.npy', X_train)
	np.save(kwargs['save_path']+'hwe_data/hwe_X_val.npy', X_test)
	np.save(kwargs['save_path']+'hwe_data/hwe_y_train.npy', y_train)
	np.save(kwargs['save_path']+'hwe_data/hwe_y_val.npy', y_test)
	
	# except Exception:
	# 	import traceback
	# 	print(traceback.format_exc())


def data_gen_process_vlv(*args, **kwargs):
	
	# read the data from the database
	df = kwargs['df'].copy()


	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0, inplace=True) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_mean(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)


	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']


	# add binary classification column
	df['vlv'] = 1.0
	df.loc[df['hwe']<= 0.001, ['vlv']] = 0


	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(weeks=10)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_test, X_train, y_test, y_train = a_utils.df_2_arrays(df = df,
		predictorcols = ['oat', 'oah', 'wbt', 'sat-oat'], outputcols = ['vlv'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)


	# save test ids for later plots
	# idx_end = -max(X_test.shape[1],y_test.shape[1])
	# idx_start = idx_end - X_test.shape[0] + 1
	# test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	# test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	# with open(kwargs['save_path']+'vlv_data/vlv_test_info.txt', 'a') as ifile:
	# 	ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'vlv_data/vlv_X_train.npy', X_train)
	np.save(kwargs['save_path']+'vlv_data/vlv_X_val.npy', X_test)
	np.save(kwargs['save_path']+'vlv_data/vlv_y_train.npy', y_train)
	np.save(kwargs['save_path']+'vlv_data/vlv_y_val.npy', y_test)


def data_gen_process_env(*args, **kwargs):

	# read the data from the database
	df = kwargs['df'].copy()

	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0, inplace=True) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_mean(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)

	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']

	# create avg_stpt column
	stpt_cols = [ele for ele in df.columns if 'vrf' in ele]
	df['avg_stpt'] = df[stpt_cols].mean(axis=1)
	# drop individual set point cols
	df.drop( columns = stpt_cols, inplace = True)

	# select retrain range of the data
	time_start_of_train = df.index[-1]-timedelta(weeks=kwargs['retrain_range_rl_weeks'])
	df = df.loc[time_start_of_train : , :]

	# save the data frame
	df.to_pickle(kwargs['save_path']+'env_data/env_data.pkl')


def online_data_gen(*args, **kwargs):
	"""
	This process should trigger data collection and correspondingly trigger other lstm and agent
	relearning modules based on current time being at the end of a regular interval of whatever(or error
	tracking measure). Also, it should use the bdx api to get the raw data-> clean and process it 
	using data_process -> and finally do the data_gen_* methods which will trigger model learning
	which is followed by model training
	"""
	
	# logger
	log = kwargs['logger']
	try: 
		# Events
		lstm_data_available : Event = kwargs['lstm_data_available']  # new data available for lstm relearning
		env_data_available : Event = kwargs['env_data_available']  # new data available for env relearning  # pylint: disable=unused-variable
		end_learning : Event = kwargs['end_learning'] 

		# Locks
		lstm_train_data_lock : Lock = kwargs['lstm_train_data_lock']  # prevent dataloop from writing data
		env_train_data_lock : Lock = kwargs['env_train_data_lock']  # prevent dataloop from writing data  # pylint: disable=unused-variable

		# relearn interval in date time format- first relearn happends fast
		relearn_interval_kwargs = {'days':0, 'hours':0, 'minutes':0, 'seconds':0}  # eg {'days':6, 'hours':23, 'minutes':50, 'seconds':0}
		# retrain range in weeks
		retrain_range_weeks = kwargs['retrain_range_weeks']
		# rl retrain weeks
		retrain_range_rl_weeks = kwargs['retrain_range_rl_weeks']

		# time at which this method started: used for book triggering relearn loop
		last_train_time = datetime.now()
		year_num, week_num, _ = last_train_time.isocalendar()

		#get auth api and meta data dictionary
		with open('auths.json', 'r') as fp:
			api_args = json.load(fp)
		with open('alumni_scripts/meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)

		while not end_learning.is_set():  # query data at regular intervals

			# condition 1 : data availability -- set to True to be always satisifed
			data_unavailable = not (lstm_data_available.is_set() | env_data_available.is_set())
			# condition 2 : interval satisfied -- set to True to be always satisifed
			interval_completed = (datetime.now()-last_train_time) > timedelta(**relearn_interval_kwargs)
			relearn_interval_kwargs = kwargs['relearn_interval_kwargs']
			# condition 3 : some error is crossing a threshold -- set to True to be always satisifed
			error_trigger = True
			# condition 4 : some reward measure is crossing a threshold -- set to True to be always satisifed
			reward_trigger = True

			if data_unavailable & interval_completed & error_trigger & reward_trigger:

				# update last_train_time
				last_train_time = datetime.now()

				# get the new data
				log.info('OnlineDataGen: Getting Data from BdX')
				df = get_train_data(api_args, meta_data_, retrain_range_weeks, log)

				# Thread for data preparation
				data_gen_process_cwe_th = Thread(target=data_gen_process_cwe, daemon=False,
											kwargs={ 
											'df' : df.loc[:,kwargs['cwe_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_hwe_th = Thread(target=data_gen_process_hwe, daemon=False, 
											kwargs={ 
											'df' : df.loc[:,kwargs['hwe_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_vlv_th = Thread(target=data_gen_process_vlv, daemon=False, 
											kwargs={ 
											'df' : df.loc[:,kwargs['vlv_vars']],
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
											'week_num': week_num, 'save_path':kwargs['save_path'] 
											})
				data_gen_process_env_th = Thread(target=data_gen_process_env, daemon=False, 
											kwargs={
											'df' : df,
											'agg': kwargs['agg'], 'scaler': kwargs['scaler'],
											'save_path':kwargs['save_path'], 'retrain_range_rl_weeks':retrain_range_rl_weeks
											})
				with lstm_train_data_lock:
					data_gen_process_cwe_th.start()
					data_gen_process_hwe_th.start()
					data_gen_process_vlv_th.start()
					data_gen_process_cwe_th.join()
					data_gen_process_vlv_th.join()
					data_gen_process_hwe_th.join()
				lstm_data_available.set()  # data is now available for lstm training
			
				with env_train_data_lock:	
					data_gen_process_env_th.start()
					data_gen_process_env_th.join()
				env_data_available.set()  # data is now available for agent and env training

				log.info('OnlineDataGen: Dynamic Model and Gym Env data available')
				# with lstm_train_data_lock:
				# 	data_gen_process_cwe(**{ 'df' : df.loc[:,kwargs['cwe_vars']],
				# 							'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
				# 							'week_num': week_num, 'save_path':kwargs['save_path']})
				# 	data_gen_process_hwe(**{ 'df' : df.loc[:,kwargs['hwe_vars']],
				# 							'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
				# 							'week_num': week_num, 'save_path':kwargs['save_path']})
				# 	data_gen_process_vlv(**{ 'df' : df.loc[:,kwargs['vlv_vars']],
				# 							'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
				# 							'week_num': week_num, 'save_path':kwargs['save_path']})
				# lstm_data_available.set()
				# with env_train_data_lock:
				# 	data_gen_process_env(**{'df' : df,
				# 							'agg': kwargs['agg'], 'scaler': kwargs['scaler'],
				# 							'save_path':kwargs['save_path']})		
				# env_data_available.set()
				


				week_num += 1
				week_num = week_num if week_num%53 != 0 else 1
				year_num = year_num if week_num!= 1 else year_num+1

			# else:	
				# sleep for 5 minutes to prevent fast loops in case relearn interval is large and 
				# other conditions have not been satisfied
				# time.sleep(timedelta(minutes=1).seconds)
	except Exception as e:
		log.error('On-Line Data Generator Module: %s', str(e))
		log.debug(e, exc_info=True)
	

def get_train_data(api_args, meta_data_, retrain_range_weeks, log):

	try:
		# arguements for the api query
		time_args = {'trend_id' : '2681', 'save_path' : 'data/trend_data/alumni_data_train.csv'}
		start_fields = ['start_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
		end_fields = ['end_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
		end_time = datetime.now(tz=pytz.utc)
		start_time = end_time - timedelta(weeks=retrain_range_weeks)
		for idx, i in enumerate(start_fields):
			time_args[i] = start_time.timetuple()[idx]
		for idx, i in enumerate(end_fields):
			time_args[i] = end_time.timetuple()[idx]
		api_args.update(time_args)

		# pull the data into csv file
		try:
			dp.pull_offline_data(**api_args)
			log.info('OnlineDataGen: Train Data Obtained  using API')
		except Exception:
			log.info('OnlineDataGen: BdX API could not get train data: will resuse old data')

		# get the dataframe from a csv
		while True:
			if not os.path.exists('data/trend_data/alumni_data_train_wbt.csv'):
				log.info('OnlineDataGen: Start of Wet Bulb Data Calculation; wait 40 s')
				time.sleep(timedelta(seconds=40).seconds)
			else:
				log.info('OnlineDataGen: Wet Bulb Data Calculation is almost done, wait 40 s')
				time.sleep(timedelta(seconds=40).seconds) # give some time to finish writing
				break

		df_ = read_csv('data/trend_data/alumni_data_train_wbt.csv', )
		os.remove('data/trend_data/alumni_data_train_wbt.csv')
		df_['time'] = to_datetime(df_['time'])
		to_zone = tz.tzlocal()
		df_['time'] = df_['time'].apply(lambda x: x.astimezone(to_zone)) # convert time to loca timezones
		df_.set_index(keys='time',inplace=True, drop = True)
		df_ = a_utils.dropNaNrows(df_)

		# add wet bulb temperature to the data
		#log.info('OnlineDataGen: Start of Wet Bulb Data Calculation')
		#rh = df_['WeatherDataProfile humidity']/100
		#rh = rh.to_numpy()
		#t_db = 5*(df_['AHU_1 outdoorAirTemp']-32)/9 + 273.15
		#t_db = t_db.to_numpy()

		# tdb_rh = np.concatenate((t_db.reshape(-1,1), rh.reshape(-1,1)), axis=1)
		# chunks = [ (sub_arr[:, 0].flatten(), sub_arr[:, 1].flatten(), cpu_id)
		# 			for cpu_id, sub_arr in enumerate(np.array_split(tdb_rh, multiprocessing.cpu_count(), axis=0))]
		# pool = multiprocessing.Pool()
		# individual_results = pool.map(calculate_wbt, chunks)
		# # Freeing the workers:
		# pool.close()
		# pool.join()
		# T = np.concatenate(individual_results)

		#T = HAPropsSI('T_wb','R',rh,'T',t_db,'P',101325)
		#t_f = 9*(T-273.15)/5 + 32
		#df_['wbt'] = t_f
		# log.info('OnlineDataGen: Wet Bulb Data Calculated')

		# rename the columns
		new_names = []
		for i in df_.columns:
			new_names.append(meta_data_["reverse_col_alias"][i])
		df_.columns = new_names

		# clean the data
		df_cleaned = dp.offline_batch_data_clean(
			meta_data_ = meta_data_, df = df_
		)

		return df_cleaned
	
	except Exception as e:
		log.error('Date Generator Get Online Train Data Module: %s', str(e))
		log.debug(e, exc_info=True)


def calculate_wbt(all_args):
	t_db, rh, cpu_id = all_args
	proc = psutil.Process()
	proc.cpu_affinity([cpu_id])
	T = HAPropsSI('T_wb','R',rh,'T',t_db,'P',101325)
	return T