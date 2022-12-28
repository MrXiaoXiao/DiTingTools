"""
# DiTing Foreshock Detection and Analysis auto-workflow
# Usage: python main.py --config-file [path to config files]
"""

from sqlite3 import DataError
import time
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
import obspy
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import yaml
import argparse
import os
from pathlib import Path
from src.PostProcessing import postprocesser_ev_center
import json
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed

def append_stadict_to_picks_csv(append_csv_path, sta_dict, fname):
    if os.path.exists(append_csv_path):
        csv_file = pd.read_csv(append_csv_path)
    else:
        csv_file = pd.DataFrame(columns=['fname','starttime','itp','tp_prob','its','ts_prob','p_polarity','p_snr'])
    for key in sta_dict.keys():
        for ev_info in sta_dict[key]:
            starttime = obspy.UTCDateTime(ev_info[7])
            try:
                itp = [ int(100*(obspy.UTCDateTime(ev_info[0])- starttime)) ]
            except:
                continue
            tp_prob = [ ev_info[1] ]
            try:
                its = [ int(100*( obspy.UTCDateTime(ev_info[2])- starttime)) ]
            except:
                its = []
            ts_prob = [ ev_info[3] ]
            p_polarity = [ev_info[4]]
            p_snr = ev_info[6]
            csv_file.loc[len(csv_file.index)] = [fname + key,starttime,itp,tp_prob,its,ts_prob,p_polarity,p_snr]
    csv_file.to_csv(append_csv_path, index=False)
    return

def process_sta_list(input_dict):
    GPU_ID = input_dict['GPU_ID']
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    import warnings
    warnings.filterwarnings("ignore")
    st = input_dict['st'] 
    sta_dict = input_dict['sta_dict']
    fname = input_dict['fname']
    print('Loading Picker Model...This might take a while...')
    #inv = obspy.read_inventory('./YC_MY_9G_station.xml')
    inv = None
    pad_length = 0
    paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1000,
              'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}
    picker_model_path = input_dict['picker_model_path']
    picker_model = tf.keras.models.load_model(picker_model_path, compile=False)

    print('Loading First Motion Polarity Model...This might take a while...')
    motion_model_path = input_dict['motion_model_path']
    motion_model = tf.keras.models.load_model(motion_model_path, compile=False)

    print('Loading Earthquake Magnitude Model...This might take a while...')
    mag_model_path = input_dict['mag_model_path']
    mag_model = tf.keras.models.load_model(mag_model_path, compile=False)

    sta_dict_key_list = input_dict['sta_dict_key_list'] 
    """
    IMPORTANT PARAMS
    """
    save_plot = True
    apply_filter = True
    waveform_folder = input_dict['waveform_folder']
    plot_folder = input_dict['plot_folder']
    show_mode = input_dict['show_mode']
    ############################################################
    """
    USUALLY DONT NEED TO CHANGE
    """
    input_N = 6144
    slide_step = 1000
    test_batch_size = 20
    det_th = 0.3
    p_th = 0.10
    s_th = 0.01
    p_mpd = 400
    s_mpd = 800
    ev_tolerance = 80
    ps_min_tolerance = 80
    p_tolerance = 200
    s_tolerance = 400
    eq_center_cluster_range = 1000
    mini_cluster_num = 4
    mini_pick_num = 4
    # max_prob or median
    phase_keep_rule = 'max_prob'
    ############################################################
    # define dist function
    dist = lambda p1, p2: np.abs(p1-p2)

    pad_length = 0
    paz_wa = {'sensitivity': 1, 'zeros': [0j], 'gain': 1,
              'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}
    for net_sta_key_index in range(len(sta_dict_key_list)):
        net_sta_key = sta_dict_key_list[net_sta_key_index]
        net = net_sta_key.split('.')[0]
        sta = net_sta_key.split('.')[1]
        temp_st_ori = st.select(network=net,station=sta)
        temp_st = temp_st_ori.copy() 
        if len(temp_st) == 0:
            continue
        if temp_st[0].stats.sampling_rate != 100.0:
            print('Resampling...')
            temp_st.resample(100.0)
        if apply_filter:
            try:
                temp_st.filter('bandpass',freqmin=0.2,freqmax=15,zerophase=True)
                temp_st.taper(0.05)
            except:
                continue
        
        temp_st.trim(temp_st[0].stats.starttime - pad_length, temp_st[0].stats.endtime + pad_length, pad=True, fill_value=0.0)
        st_len = len(temp_st[0].data)
        
        slice_num = int( (st_len-input_N+slide_step) / slide_step)
        if slice_num < 0:
            continue
        total_batch_number = int(slice_num/test_batch_size) + 1

        for batch_num in range(total_batch_number):
            temp_input = np.zeros([test_batch_size,input_N,3])
            P_pick_list = []
            S_pick_list = []
            ev_centers  = []
            for batch_dx in range(test_batch_size):
                slice_dx = batch_num*test_batch_size + batch_dx
                for trace in temp_st.traces:
                    end_dx = input_N
                    end_dx_tx = slice_dx*slide_step + input_N
                    if  slice_dx*slide_step + input_N > len(trace.data):
                        end_dx = input_N - (slice_dx*slide_step + input_N - len(trace.data))
                        end_dx_tx = len(trace.data)
                        if end_dx < 2000:
                            continue

                    if trace.stats.channel[2] == 'Z':
                        temp_input[batch_dx,:end_dx,0] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,0]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,0] -= np.mean(temp_input[batch_dx,:end_dx,0])
                            norm_factor = np.std(temp_input[batch_dx,:,0])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,0] /= norm_factor
                            
                    elif trace.stats.channel[2] == 'N' or trace.stats.channel[2] == '1':
                        temp_input[batch_dx,:end_dx,1] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,1]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,1] -= np.mean(temp_input[batch_dx,:end_dx,1])
                            norm_factor = np.std(temp_input[batch_dx,:,1])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,1] /= norm_factor
                            
                    elif trace.stats.channel[2] == 'E' or trace.stats.channel[2] == '2':
                        temp_input[batch_dx,:end_dx,2] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,2]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,2] -= np.mean(temp_input[batch_dx,:end_dx,2])
                            norm_factor = np.std(temp_input[batch_dx,:,2])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,2] /= norm_factor
            
            try:
                pred_res = picker_model.predict(temp_input)
            except:
                print('Error on data {}'.format(net_sta_key))
                continue
            if show_mode:
                show_list = list()
            for batch_dx in range(test_batch_size):
                slice_dx = batch_num*test_batch_size + batch_dx
                P_pred_prob = pred_res['C0D3'][batch_dx,0:input_N,0]
                S_pred_prob = pred_res['C1D3'][batch_dx,0:input_N,0]
                D_pred_prob = pred_res['C2D3'][batch_dx,0:input_N,0]

                post_res = postprocesser_ev_center(D_pred_prob,
                                                    P_pred_prob,
                                                    S_pred_prob,
                                                    det_th = det_th,
                                                    p_th = p_th,
                                                    s_th = s_th,
                                                    p_mpd = p_mpd,
                                                    s_mpd = s_mpd,
                                                    ev_tolerance = ev_tolerance,
                                                    p_tolerance = p_tolerance,
                                                    s_tolerance = s_tolerance)

                for t_res in post_res:
                    ev_centers.append(t_res[0] + slice_dx*slide_step)
                    P_pick_array = np.asarray(t_res[1],dtype=np.float64)
                    P_pick_array[:,0] += slice_dx*slide_step
                    P_pick_list.append(P_pick_array)
                    
                    S_pick_array = np.asarray(t_res[2],dtype=np.float64)
                    S_pick_array[:,0] += slice_dx*slide_step
                    S_pick_list.append(S_pick_array)
            
            # Eq detection results agglomerative clustering
            if len(ev_centers) <= mini_cluster_num:
                continue
            else:
                dm = np.asarray([[dist(e1, e2) for e2 in ev_centers] for e1 in ev_centers])
                cluster_res = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=eq_center_cluster_range).fit(dm)
                P_median_dict = dict()
                S_median_dict = dict()
                
                #print(cluster_res.labels_)

                for cluster_dx in range(len(cluster_res.labels_)):
                    #print(cluster_dx)
                    label_key = cluster_res.labels_[cluster_dx]
                    #print(label_key)
                    if label_key not in P_median_dict.keys():
                        P_median_dict[label_key] = list()
                    P_median_dict[label_key].append(P_pick_list[cluster_dx])
                    if label_key not in S_median_dict.keys():
                        S_median_dict[label_key] = list()
                    S_median_dict[label_key].append(S_pick_list[cluster_dx])
                
                pre_P = None

                for pick_key in P_median_dict.keys():
                    temp_Ps = list(itertools.chain(*P_median_dict[pick_key]))
                    total_pick_num = len(temp_Ps)
                    if total_pick_num < mini_pick_num:
                        append_P_pick = None
                        append_P_prob = None
                    else:
                        temp_Ps = np.reshape(np.asarray(temp_Ps, dtype=np.float64), [total_pick_num, 2])
                        if phase_keep_rule == 'max_prob':
                            max_index = np.argmax(temp_Ps[:,1])
                            append_P_pick = temp_Ps[max_index,0]
                            append_P_prob = temp_Ps[max_index,1]
                        else:
                            # get median index
                            median_index = np.argsort(temp_Ps[:,0])[len(temp_Ps)//2]
                            append_P_pick = temp_Ps[median_index,0]
                            append_P_prob = temp_Ps[median_index,1]
                        if np.isnan(append_P_pick): 
                            continue
                        else:
                            append_P_pick = temp_st[0].stats.starttime + append_P_pick*0.01
                    
                    if append_P_pick is None:
                        continue

                    if pre_P is None:
                        pre_P = append_P_pick
                    elif np.abs(pre_P - append_P_pick) < 2.0:
                        continue
                    else:
                        pre_P = append_P_pick

                    # get median index
                    temp_Ss = list(itertools.chain(*S_median_dict[pick_key]))
                    total_pick_num = len(temp_Ss)
                    if total_pick_num < mini_pick_num:
                        append_S_pick = None
                        append_S_prob = None
                    else:
                        temp_Ss = np.reshape(np.asarray(temp_Ss, dtype=np.float64), [total_pick_num, 2])
                        median_index = np.argsort(temp_Ss[:,0])[len(temp_Ss[:,0])//2]
                        append_S_pick = temp_Ss[median_index,0]
                        if np.isnan(append_S_pick):
                            append_S_pick = None
                            append_S_prob = None
                        else:
                            append_S_pick = temp_st[0].stats.starttime + append_S_pick*0.01
                            append_S_prob = temp_Ss[median_index,1]
                            if append_S_pick - append_P_pick < ps_min_tolerance/100.0:
                                append_S_pick = None
                                append_S_prob = None
                    # perform motion detection
                    # first motion polarity
                    temp_st = st.select(network=net,station=sta)
                    keep_P = int( (append_P_pick - temp_st[0].stats.starttime)*100 )
                    if append_S_pick:
                        psdiff = append_S_pick - append_P_pick
                        if psdiff > input_dict['max_psdiff'] or psdiff < input_dict['min_psdiff'] :
                            continue
                        distance = psdiff * 8
                    else:
                        distance = None

                    # calculate the magnitude
                    if distance != None:
                        try:
                            temp_mag = temp_st_ori.copy()
                            temp_mag.trim(append_S_pick - 10.0, append_S_pick + 40.0)
                            temp_mag.detrend()
                            pre_filt = [0.001, 0.005, 45, 50]
                            temp_mag.remove_response(inventory=inv, output="DISP", pre_filt = pre_filt )
                            temp_mag.simulate(paz_simulate=paz_wa)
                            temp_mag.filter('bandpass',freqmin=0.2,freqmax=10.0,zerophase=True)
                            temp_mag.trim(append_S_pick - 5, append_S_pick + 20)
                            max_n = max(temp_mag.select(channel='??N')[0].data)
                            max_e = max(temp_mag.select(channel='??E')[0].data)
                            amp = (max_n + max_e)/2.0
                            ml = np.log(amp) / np.log(10) + 1.110 * np.log(distance / 100) / np.log(10) + 0.00189 * (distance - 100) + 3.5
                        except:
                            ml = None
                    else:
                        ml = None
                    ml = str(ml)

                    starttime_csv_usage = temp_st[0].stats.starttime
                    p_snr = 0
                    motion_input = np.zeros([1,128,2])
                    has_Z_channel = False

                    for trace in temp_st.traces:
                        if trace.stats.channel[2] == 'Z':
                            try:
                                motion_input[0,:,0] = trace.data[keep_P-64:keep_P + 64]
                                p_snr = np.max( np.abs(trace.data[keep_P:keep_P + 50]) )/ np.max(np.abs(trace.data[keep_P-50:keep_P]) )
                            except:
                                continue
                            if np.max(motion_input[0,:,0]) == 0:
                                pass
                            else:
                                has_Z_channel = True
                                motion_input[0,:,0] -= np.mean(motion_input[0,:,0])
                                norm_factor = np.max(np.abs(motion_input[0,:,0]))
                                if norm_factor == 0:
                                    pass
                                else:
                                    motion_input[0,:,0] /= norm_factor
                                diff_data = np.diff(motion_input[0,64:,0])
                                diff_sign_data = np.sign(diff_data)
                                motion_input[0,64+1:,1] = diff_sign_data[:]

                    if has_Z_channel:
                        # remove artifical boundary
                        if np.sum(np.abs(np.diff(motion_input[0,5:50,0]))) == 0:
                            # print('Skipping Zero bound')
                            continue

                        motion_res_all = motion_model.predict(motion_input)
                        motion_res = motion_res_all['T0D3']
                        
                        if np.argmax(motion_res[0,:]) == 0:
                            polarity = 'U'
                        elif np.argmax(motion_res[0,:]) == 1:
                            polarity = 'D'
                        else:
                            polarity = '-'
                            
                        motion_sharpness = motion_res_all['T1D3']
                        if np.argmax(motion_sharpness[0,:]) == 0:
                            sharpness = 'I'
                        elif np.argmax(motion_sharpness[0,:]) == 1:
                            sharpness = 'E'
                        else:
                            sharpness = '-'
                    else:
                        polarity = None
                        sharpness = None
                        # print('Skipping No Z channel')
                        continue
                    
                    # AI magnitude reg
                    mag_input = np.zeros([1,1024,3])
                    for trace in temp_st.traces:
                        if trace.stats.channel[2] == 'Z':
                            try:
                                mag_input[0,:,0] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,0] -= np.mean(mag_input[0,:,0])
                            except:
                                pass
                        if trace.stats.channel[2] == 'N':
                            try:
                                mag_input[0,:,1] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,1] -= np.mean(mag_input[0,:,1])
                            except:
                                pass
                        if trace.stats.channel[2] == 'E':
                            try:
                                mag_input[0,:,2] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,2] -= np.mean(mag_input[0,:,2])
                            except:
                                pass

                    #ml_ai_pred = mag_model.predict(mag_input)
                    ml_ai = -999
                    sta_dict[net_sta_key].append([str(append_P_pick), str(append_P_prob), str(append_S_pick), str(append_S_prob), polarity, sharpness, str(p_snr), str(starttime_csv_usage),str(ml_ai),str(ml)])

                    if show_mode:
                        P_pick_proj_back = int(100*(append_P_pick - temp_st[0].stats.starttime))
                        try:
                            S_pick_proj_back = int(100*(append_S_pick - temp_st[0].stats.starttime))
                        except:
                            S_pick_proj_back = -9999
                        show_list.append([P_pick_proj_back, S_pick_proj_back, polarity, sharpness, p_snr])
                    
                    if os.path.exists(waveform_folder + '{}/'.format(fname)):
                        pass
                    else:
                        os.makedirs(waveform_folder + '{}/'.format(fname))
                    save_st = temp_st_ori.copy().trim(starttime=append_P_pick - 5.0, endtime = append_P_pick + 10.0)
                    save_st.write(waveform_folder + '{}/{}.EV.{}.mseed'.format(fname,net_sta_key,len(sta_dict[net_sta_key])))                    
                    
                    # save plot
                    if save_plot:
                        #print('Plotting')
                        short_mark = False
                        if os.path.exists(plot_folder + '{}/'.format(fname)):
                            pass
                        else:
                            os.makedirs(plot_folder + '{}/'.format(fname))
                        temp_st = st.select(network=net,station=sta)
                        plt.figure(figsize=(5,3))
                        tdx = 0
                        
                        if append_P_pick is None:
                            continue
                        else:
                            P_plot = (append_P_pick - temp_st[0].stats.starttime)*100
                        
                        if append_S_pick is None:
                            S_plot = (append_P_pick - temp_st[0].stats.starttime)*100
                        else:
                            S_plot = (append_S_pick - temp_st[0].stats.starttime)*100
                        
                        for trace in temp_st.traces:
                            try:
                                plot_data = trace.data[int(P_plot-300):int(S_plot+2000)] - np.mean(trace.data[int(P_plot-300):int(S_plot+2000)])
                                plot_data /= np.max(np.abs(plot_data))
                            except:
                                try:
                                    plot_data = trace.data[int(P_plot-50):int(P_plot+50)] - np.mean(trace.data[int(P_plot-50):int(P_plot+50)])
                                    plot_data /= np.max(np.abs(plot_data))
                                    short_mark = True
                                except:
                                    plt.close()
                                    continue
                            plt.plot(plot_data + tdx*2,color='k')
                            tdx += 1
                        if append_P_pick is not None: 
                            if short_mark:
                                plt.plot([50, 50],[-1,5],color='b',linestyle='--', label='DiTing P pick')
                            else:
                                plt.plot([300, 300],[-1,5],color='b',linestyle='--', label='DiTing P pick')
                        
                        if append_S_pick is not None: 
                            plt.plot([S_plot-P_plot+300, S_plot-P_plot+300],[-1,5],color='r',linestyle='--', label='DiTing S pick')
                        
                        plt.title('P pick time: {}\nP prob: {}\nS pick time: {}\nS prob: {}\nPolarity: {}  Sharpness: {} Ml: {}'.format(append_P_pick, append_P_prob, append_S_pick, append_S_prob, polarity, sharpness, ml))
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(plot_folder + '{}/{}.EV.{}.jpg'.format(fname,net_sta_key,len(sta_dict[net_sta_key])),dpi=100)
                        plt.close()
    
            
            if show_mode:
                for batch_dx in range(test_batch_size):
                    if (batch_dx % batch_show_gap) == 0:
                        slice_dx = batch_num*test_batch_size + batch_dx
                        trace_array = np.ndarray(shape=(6144,3))
                        trace_array[:,:] = temp_input[batch_dx, :, :]

                        if np.max(np.abs(trace_array)) < 0.1:
                            continue
                        
                        P_pred_prob = pred_res['C0D3'][batch_dx,0:input_N,0]
                        S_pred_prob = pred_res['C1D3'][batch_dx,0:input_N,0]
                        D_pred_prob = pred_res['C2D3'][batch_dx,0:input_N,0]
                        
                        P_appearance_array = np.zeros(6144)
                        S_appearance_array = np.zeros(6144)
                        
                        polarity_show_list = list()
                        for _ in range(6144):
                            polarity_show_list.append(0)
                        
                        sharpness_show_list = list()
                        for _ in range(6144):
                            sharpness_show_list.append(0)

                        polar_snr_list = np.zeros(6144)

                        for show_info in show_list:
                            if show_info[0] > slice_dx*slide_step and show_info[0] < slice_dx*slide_step + input_N:
                                P_appear_idx = show_info[0] - slice_dx*slide_step
                                P_appearance_array[P_appear_idx] = 1
                                polarity_show_list[P_appear_idx] = show_info[2]
                                sharpness_show_list[P_appear_idx] = show_info[3]
                                polar_snr_list[P_appear_idx] = show_info[4]
                            if show_info[1] > slice_dx*slide_step and show_info[1] < slice_dx*slide_step + input_N:
                                S_appear_idx = show_info[1] - slice_dx*slide_step
                                S_appearance_array[S_appear_idx] = 1
                        
                        P_appearance_array = P_appearance_array.reshape([6144,1])
                        S_appearance_array = S_appearance_array.reshape([6144,1])
                        sharpness_show_list = np.asarray(sharpness_show_list).reshape([6144,1])
                        polarity_show_list = np.asarray(polarity_show_list).reshape([6144,1])
                        polar_snr_list = polar_snr_list.reshape([6144,1])

                        save_data = np.hstack( (trace_array[:6000,:], P_pred_prob.reshape([6144,1])[:6000,:], S_pred_prob.reshape([6144,1])[:6000,:], D_pred_prob.reshape([6144,1])[:6000,:], P_appearance_array[:6000,:], S_appearance_array[:6000,:], polarity_show_list[:6000,:], polar_snr_list[:6000,:], sharpness_show_list[:6000,:]) )
                        sub_dir_name = './' + fname + '_show_csv/'
                        if os.path.exists(sub_dir_name):
                            pass
                        else:
                            os.makedirs(sub_dir_name)
                        np.savetxt(sub_dir_name + sta + fname + '({}).csv'.format(batch_dx), save_data, delimiter=',', fmt='%s')
    return sta_dict

def process_sta_list_passing_model(input_dict, picker_model, motion_model, mag_model):
    st = input_dict['st'] 
    sta_dict = input_dict['sta_dict']
    fname = input_dict['fname']
    sta_dict_key_list = input_dict['sta_dict_key_list'] 
    """
    IMPORTANT PARAMS
    """
    save_plot = True
    apply_filter = True
    waveform_folder = input_dict['waveform_folder']
    plot_folder = input_dict['plot_folder']
    show_mode = input_dict['show_mode']
    ############################################################
    """
    USUALLY DONT NEED TO CHANGE
    """
    input_N = 6144
    slide_step = 1000
    test_batch_size = 20
    det_th = 0.3
    p_th = 0.10
    s_th = 0.01
    p_mpd = 400
    s_mpd = 800
    ev_tolerance = 80
    ps_min_tolerance = 80
    p_tolerance = 200
    s_tolerance = 400
    eq_center_cluster_range = 1000
    mini_cluster_num = 4
    mini_pick_num = 4
    # max_prob or median
    phase_keep_rule = 'max_prob'
    ############################################################
    # define dist function
    dist = lambda p1, p2: np.abs(p1-p2)

    #inv = obspy.read_inventory('./YC_MY_9G_station.xml')
    inv = None
    pad_length = 0
    paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1000,
              'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}
    
    for net_sta_key_index in range(len(sta_dict_key_list)):
        net_sta_key = sta_dict_key_list[net_sta_key_index]
        net = net_sta_key.split('.')[0]
        sta = net_sta_key.split('.')[1]
        temp_st_ori = st.select(network=net,station=sta)
        temp_st = temp_st_ori.copy() 
        if len(temp_st) == 0:
            continue
        if temp_st[0].stats.sampling_rate != 100.0:
            print('Resampling...')
            temp_st.resample(100.0)
        if apply_filter:
            try:
                temp_st.filter('bandpass',freqmin=0.2,freqmax=15,zerophase=True)
                temp_st.taper(0.05)
            except:
                continue
        
        temp_st.trim(temp_st[0].stats.starttime - pad_length, temp_st[0].stats.endtime + pad_length, pad=True, fill_value=0.0)
        st_len = len(temp_st[0].data)
        
        slice_num = int( (st_len-input_N+slide_step) / slide_step)
        if slice_num < 0:
            continue
        total_batch_number = int(slice_num/test_batch_size) + 1

        for batch_num in range(total_batch_number):
            temp_input = np.zeros([test_batch_size,input_N,3])
            P_pick_list = []
            S_pick_list = []
            ev_centers  = []
            for batch_dx in range(test_batch_size):
                slice_dx = batch_num*test_batch_size + batch_dx
                for trace in temp_st.traces:
                    end_dx = input_N
                    end_dx_tx = slice_dx*slide_step + input_N
                    if  slice_dx*slide_step + input_N > len(trace.data):
                        end_dx = input_N - (slice_dx*slide_step + input_N - len(trace.data))
                        end_dx_tx = len(trace.data)
                        if end_dx < 2000:
                            continue

                    if trace.stats.channel[2] == 'Z':
                        temp_input[batch_dx,:end_dx,0] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,0]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,0] -= np.mean(temp_input[batch_dx,:end_dx,0])
                            norm_factor = np.std(temp_input[batch_dx,:,0])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,0] /= norm_factor
                            
                    elif trace.stats.channel[2] == 'N' or trace.stats.channel[2] == '1':
                        temp_input[batch_dx,:end_dx,1] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,1]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,1] -= np.mean(temp_input[batch_dx,:end_dx,1])
                            norm_factor = np.std(temp_input[batch_dx,:,1])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,1] /= norm_factor
                            
                    elif trace.stats.channel[2] == 'E' or trace.stats.channel[2] == '2':
                        temp_input[batch_dx,:end_dx,2] = trace.data[slice_dx*slide_step:end_dx_tx]
                        if np.max(temp_input[batch_dx,:,2]) == 0:
                            pass
                        else:
                            temp_input[batch_dx,:end_dx,2] -= np.mean(temp_input[batch_dx,:end_dx,2])
                            norm_factor = np.std(temp_input[batch_dx,:,2])
                            if norm_factor == 0:
                                pass
                            else:
                                temp_input[batch_dx,:,2] /= norm_factor
            
            try:
                pred_res = picker_model.predict(temp_input)
            except:
                print('Error on data {}'.format(net_sta_key))
                continue
            if show_mode:
                show_list = list()
            for batch_dx in range(test_batch_size):
                slice_dx = batch_num*test_batch_size + batch_dx
                P_pred_prob = pred_res['C0D3'][batch_dx,0:input_N,0]
                S_pred_prob = pred_res['C1D3'][batch_dx,0:input_N,0]
                D_pred_prob = pred_res['C2D3'][batch_dx,0:input_N,0]

                post_res = postprocesser_ev_center(D_pred_prob,
                                                    P_pred_prob,
                                                    S_pred_prob,
                                                    det_th = det_th,
                                                    p_th = p_th,
                                                    s_th = s_th,
                                                    p_mpd = p_mpd,
                                                    s_mpd = s_mpd,
                                                    ev_tolerance = ev_tolerance,
                                                    p_tolerance = p_tolerance,
                                                    s_tolerance = s_tolerance)

                for t_res in post_res:
                    ev_centers.append(t_res[0] + slice_dx*slide_step)
                    P_pick_array = np.asarray(t_res[1],dtype=np.float64)
                    P_pick_array[:,0] += slice_dx*slide_step
                    P_pick_list.append(P_pick_array)
                    
                    S_pick_array = np.asarray(t_res[2],dtype=np.float64)
                    S_pick_array[:,0] += slice_dx*slide_step
                    S_pick_list.append(S_pick_array)
            
            # Eq detection results agglomerative clustering
            if len(ev_centers) <= mini_cluster_num:
                continue
            else:
                dm = np.asarray([[dist(e1, e2) for e2 in ev_centers] for e1 in ev_centers])
                cluster_res = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=eq_center_cluster_range).fit(dm)
                P_median_dict = dict()
                S_median_dict = dict()
                
                #print(cluster_res.labels_)

                for cluster_dx in range(len(cluster_res.labels_)):
                    #print(cluster_dx)
                    label_key = cluster_res.labels_[cluster_dx]
                    #print(label_key)
                    if label_key not in P_median_dict.keys():
                        P_median_dict[label_key] = list()
                    P_median_dict[label_key].append(P_pick_list[cluster_dx])
                    if label_key not in S_median_dict.keys():
                        S_median_dict[label_key] = list()
                    S_median_dict[label_key].append(S_pick_list[cluster_dx])
                
                pre_P = None

                for pick_key in P_median_dict.keys():
                    temp_Ps = list(itertools.chain(*P_median_dict[pick_key]))
                    total_pick_num = len(temp_Ps)
                    if total_pick_num < mini_pick_num:
                        append_P_pick = None
                        append_P_prob = None
                    else:
                        temp_Ps = np.reshape(np.asarray(temp_Ps, dtype=np.float64), [total_pick_num, 2])
                        if phase_keep_rule == 'max_prob':
                            max_index = np.argmax(temp_Ps[:,1])
                            append_P_pick = temp_Ps[max_index,0]
                            append_P_prob = temp_Ps[max_index,1]
                        else:
                            # get median index
                            median_index = np.argsort(temp_Ps[:,0])[len(temp_Ps)//2]
                            append_P_pick = temp_Ps[median_index,0]
                            append_P_prob = temp_Ps[median_index,1]
                        if np.isnan(append_P_pick): 
                            continue
                        else:
                            append_P_pick = temp_st[0].stats.starttime + append_P_pick*0.01

                    if append_P_pick is None:
                        continue

                    if pre_P is None:
                        pre_P = append_P_pick
                    elif np.abs(pre_P - append_P_pick) < 2.0:
                        continue
                    else:
                        pre_P = append_P_pick

                    # get median index
                    temp_Ss = list(itertools.chain(*S_median_dict[pick_key]))
                    total_pick_num = len(temp_Ss)
                    if total_pick_num < mini_pick_num:
                        append_S_pick = None
                        append_S_prob = None
                    else:
                        temp_Ss = np.reshape(np.asarray(temp_Ss, dtype=np.float64), [total_pick_num, 2])
                        median_index = np.argsort(temp_Ss[:,0])[len(temp_Ss[:,0])//2]
                        append_S_pick = temp_Ss[median_index,0]
                        if np.isnan(append_S_pick):
                            append_S_pick = None
                            append_S_prob = None
                        else:
                            append_S_pick = temp_st[0].stats.starttime + append_S_pick*0.01
                            append_S_prob = temp_Ss[median_index,1]
                            if append_S_pick - append_P_pick < ps_min_tolerance/100.0:
                                append_S_pick = None
                                append_S_prob = None
                    # perform motion detection
                    # first motion polarity
                    temp_st = st.select(network=net,station=sta)
                    keep_P = int( (append_P_pick - temp_st[0].stats.starttime)*100 )
                    if append_S_pick:
                        psdiff = append_S_pick - append_P_pick
                        if psdiff > input_dict['max_psdiff'] or psdiff < input_dict['min_psdiff'] :
                            continue
                        distance = psdiff * 8
                    else:
                        distance = None
                        
                    # calculate the magnitude
                    if distance != None:
                        try:
                            temp_mag = temp_st_ori.copy()
                            temp_mag.trim(append_S_pick - 10.0, append_S_pick + 40.0)
                            temp_mag.detrend()
                            pre_filt = [0.001, 0.005, 45, 50]
                            temp_mag.remove_response(inventory=inv, output="DISP", pre_filt = pre_filt )
                            temp_mag.simulate(paz_simulate=paz_wa)
                            temp_mag.filter('bandpass',freqmin=0.2,freqmax=10.0,zerophase=True)
                            temp_mag.trim(append_S_pick - 5, append_S_pick + 20)
                            max_n = max(temp_mag.select(channel='??N')[0].data)
                            max_e = max(temp_mag.select(channel='??E')[0].data)
                            amp = (max_n + max_e)/2.0
                            ml = np.log(amp) / np.log(10) + 1.110 * np.log(distance / 100) / np.log(10) + 0.00189 * (distance - 100) + 3.5
                        except:
                            ml = None
                    else:
                        ml = None
                    ml = str(ml)

                    starttime_csv_usage = temp_st[0].stats.starttime
                    p_snr = 0
                    motion_input = np.zeros([1,128,2])
                    has_Z_channel = False

                    for trace in temp_st.traces:
                        if trace.stats.channel[2] == 'Z':
                            try:
                                motion_input[0,:,0] = trace.data[keep_P-64:keep_P + 64]
                                p_snr = np.max( np.abs(trace.data[keep_P:keep_P + 50]) )/ np.max(np.abs(trace.data[keep_P-50:keep_P]) )
                            except:
                                continue
                            if np.max(motion_input[0,:,0]) == 0:
                                pass
                            else:
                                has_Z_channel = True
                                motion_input[0,:,0] -= np.mean(motion_input[0,:,0])
                                norm_factor = np.max(np.abs(motion_input[0,:,0]))
                                if norm_factor == 0:
                                    pass
                                else:
                                    motion_input[0,:,0] /= norm_factor
                                diff_data = np.diff(motion_input[0,64:,0])
                                diff_sign_data = np.sign(diff_data)
                                motion_input[0,64+1:,1] = diff_sign_data[:]

                    if has_Z_channel:
                        # remove artifical boundary
                        if np.sum(np.abs(np.diff(motion_input[0,5:50,0]))) == 0:
                            #print('Skipping Zero bound')
                            continue

                        motion_res_all = motion_model.predict(motion_input)
                        motion_res = motion_res_all['T0D3']

                        if np.argmax(motion_res[0,:]) == 0:
                            polarity = 'U'
                        elif np.argmax(motion_res[0,:]) == 1:
                            polarity = 'D'
                        else:
                            polarity = '-'
                            
                        motion_sharpness = motion_res_all['T1D3']
                        if np.argmax(motion_sharpness[0,:]) == 0:
                            sharpness = 'I'
                        elif np.argmax(motion_sharpness[0,:]) == 1:
                            sharpness = 'E'
                        else:
                            sharpness = '-'
                    else:
                        polarity = None
                        sharpness = None
                        #print('Skipping Zero Polar')
                        continue

                    # AI magnitude reg
                    mag_input = np.zeros([1,1024,3])
                    for trace in temp_st.traces:
                        if trace.stats.channel[2] == 'Z':
                            try:
                                mag_input[0,:,0] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,0] -= np.mean(mag_input[0,:,0])
                            except:
                                pass
                        if trace.stats.channel[2] == 'N':
                            try:
                                mag_input[0,:,1] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,1] -= np.mean(mag_input[0,:,1])
                            except:
                                pass
                        if trace.stats.channel[2] == 'E':
                            try:
                                mag_input[0,:,2] = trace.data[keep_P-100:keep_P + 924]
                                mag_input[0,:,2] -= np.mean(mag_input[0,:,2])
                            except:
                                pass

                    #ml_ai_pred = mag_model.predict(mag_input)
                    ml_ai = -999
                    sta_dict[net_sta_key].append([str(append_P_pick), str(append_P_prob), str(append_S_pick), str(append_S_prob), polarity, sharpness, str(p_snr), str(starttime_csv_usage),str(ml_ai),str(ml)])

                    # sta_dict[net_sta_key].append([str(append_P_pick), str(append_P_prob), str(append_S_pick), str(append_S_prob), polarity, sharpness, str(p_snr), str(starttime_csv_usage),str(ml)])
                    if show_mode:
                        P_pick_proj_back = int(100*(append_P_pick - temp_st[0].stats.starttime))
                        try:
                            S_pick_proj_back = int(100*(append_S_pick - temp_st[0].stats.starttime))
                        except:
                            S_pick_proj_back = -9999
                        show_list.append([P_pick_proj_back, S_pick_proj_back, polarity, sharpness, p_snr])
                    
                    if os.path.exists(waveform_folder + '{}/'.format(fname)):
                        pass
                    else:
                        os.makedirs(waveform_folder + '{}/'.format(fname))
                    save_st = temp_st_ori.copy().trim(starttime=append_P_pick - 5.0, endtime = append_P_pick + 10.0)
                    save_st.write(waveform_folder + '{}/{}.EV.{}.mseed'.format(fname,net_sta_key,len(sta_dict[net_sta_key])))                    
                    
                    # save plot
                    if save_plot:
                        #print('Plotting')
                        short_mark = False
                        if os.path.exists(plot_folder + '{}/'.format(fname)):
                            pass
                        else:
                            os.makedirs(plot_folder + '{}/'.format(fname))
                        temp_st = st.select(network=net,station=sta)
                        plt.figure(figsize=(5,3))
                        tdx = 0
                        
                        if append_P_pick is None:
                            continue
                        else:
                            P_plot = (append_P_pick - temp_st[0].stats.starttime)*100
                        
                        if append_S_pick is None:
                            S_plot = (append_P_pick - temp_st[0].stats.starttime)*100
                        else:
                            S_plot = (append_S_pick - temp_st[0].stats.starttime)*100
                        
                        for trace in temp_st.traces:
                            try:
                                plot_data = trace.data[int(P_plot-300):int(S_plot+2000)] - np.mean(trace.data[int(P_plot-300):int(S_plot+2000)])
                                plot_data /= np.max(np.abs(plot_data))
                            except:
                                try:
                                    plot_data = trace.data[int(P_plot-50):int(P_plot+50)] - np.mean(trace.data[int(P_plot-50):int(P_plot+50)])
                                    plot_data /= np.max(np.abs(plot_data))
                                    short_mark = True
                                except:
                                    plt.close()
                                    continue
                            plt.plot(plot_data + tdx*2,color='k')
                            tdx += 1
                        if append_P_pick is not None: 
                            if short_mark:
                                plt.plot([50, 50],[-1,5],color='b',linestyle='--', label='DiTing P pick')
                            else:
                                plt.plot([300, 300],[-1,5],color='b',linestyle='--', label='DiTing P pick')
                        
                        if append_S_pick is not None: 
                            plt.plot([S_plot-P_plot+300, S_plot-P_plot+300],[-1,5],color='r',linestyle='--', label='DiTing S pick')
                        
                        plt.title('P pick time: {}\nP prob: {}\nS pick time: {}\nS prob: {}\nPolarity: {}  Sharpness: {} Ml: {}'.format(append_P_pick, append_P_prob, append_S_pick, append_S_prob, polarity, sharpness, ml))
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(plot_folder + '{}/{}.EV.{}.jpg'.format(fname,net_sta_key,len(sta_dict[net_sta_key])),dpi=100)
                        plt.close()
    
            
            if show_mode:
                for batch_dx in range(test_batch_size):
                    if (batch_dx % batch_show_gap) == 0:
                        slice_dx = batch_num*test_batch_size + batch_dx
                        trace_array = np.ndarray(shape=(6144,3))
                        trace_array[:,:] = temp_input[batch_dx, :, :]

                        if np.max(np.abs(trace_array)) < 0.1:
                            continue
                        
                        P_pred_prob = pred_res['C0D3'][batch_dx,0:input_N,0]
                        S_pred_prob = pred_res['C1D3'][batch_dx,0:input_N,0]
                        D_pred_prob = pred_res['C2D3'][batch_dx,0:input_N,0]
                        
                        P_appearance_array = np.zeros(6144)
                        S_appearance_array = np.zeros(6144)
                        
                        polarity_show_list = list()
                        for _ in range(6144):
                            polarity_show_list.append(0)
                        
                        sharpness_show_list = list()
                        for _ in range(6144):
                            sharpness_show_list.append(0)

                        polar_snr_list = np.zeros(6144)

                        for show_info in show_list:
                            if show_info[0] > slice_dx*slide_step and show_info[0] < slice_dx*slide_step + input_N:
                                P_appear_idx = show_info[0] - slice_dx*slide_step
                                P_appearance_array[P_appear_idx] = 1
                                polarity_show_list[P_appear_idx] = show_info[2]
                                sharpness_show_list[P_appear_idx] = show_info[3]
                                polar_snr_list[P_appear_idx] = show_info[4]
                            if show_info[1] > slice_dx*slide_step and show_info[1] < slice_dx*slide_step + input_N:
                                S_appear_idx = show_info[1] - slice_dx*slide_step
                                S_appearance_array[S_appear_idx] = 1
                        
                        P_appearance_array = P_appearance_array.reshape([6144,1])
                        S_appearance_array = S_appearance_array.reshape([6144,1])
                        sharpness_show_list = np.asarray(sharpness_show_list).reshape([6144,1])
                        polarity_show_list = np.asarray(polarity_show_list).reshape([6144,1])
                        polar_snr_list = polar_snr_list.reshape([6144,1])

                        save_data = np.hstack( (trace_array[:6000,:], P_pred_prob.reshape([6144,1])[:6000,:], S_pred_prob.reshape([6144,1])[:6000,:], D_pred_prob.reshape([6144,1])[:6000,:], P_appearance_array[:6000,:], S_appearance_array[:6000,:], polarity_show_list[:6000,:], polar_snr_list[:6000,:], sharpness_show_list[:6000,:]) )
                        sub_dir_name = './' + fname + '_show_csv/'
                        if os.path.exists(sub_dir_name):
                            pass
                        else:
                            os.makedirs(sub_dir_name)
                        np.savetxt(sub_dir_name + sta + fname + '({}).csv'.format(batch_dx), save_data, delimiter=',', fmt='%s')
    return sta_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility of DiTing-Foreshock detection workflow')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Path to Configuration file')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file), Loader=yaml.SafeLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['GPU_ID']

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    tf.test.is_gpu_available()
    #from keras import backend as K
    #K.clear_session()

    # cfgs params
    Folder_Struct = cfgs['Folder_Struct']
    json_folder = cfgs['json_folder']
    plot_folder = cfgs['plot_folder']
    waveform_folder = cfgs['waveform_folder']
    csv_folder = cfgs['csv_folder']
    task_ID = cfgs['task_ID']
    parallel = cfgs['parallel']
    parallel_num = cfgs['parallel_num']

    if parallel:
        pass
    else:
        print('Loading Picker Model...This might take a while...')
        picker_model_path = cfgs['picker_model_path']
        picker_model = tf.keras.models.load_model(picker_model_path, compile=False)
        motion_model_path = cfgs['motion_model_path']
        motion_model = tf.keras.models.load_model(motion_model_path, compile=False)
        mag_model_path = cfgs['mag_model_path']
        mag_model = tf.keras.models.load_model(mag_model_path, compile=False)

    ############################################################
    """
    Default PARAMS
    USUALLY DONT NEED TO CHANGE
    """
    # will be slow if set True
    show_mode = False
    batch_show_gap = 3
    # define dist function
    dist = lambda p1, p2: np.abs(p1-p2)
    ############################################################
    
    #process_part_list = ['1','2','3','4']
    process_part_list = ['5']

    base_path = '/array/wuxueshan/Projects/JL-Data/{}/'

    for process_part in process_part_list:
        for sta_folder in Path(base_path.format(process_part)).glob('*'):
            for sac_file in sta_folder.glob('*'):
                if os.path.exists(json_folder + '{}_{}'.format(sta_folder.name,sac_file.name) + '.json'):
                    print('Skipping {}'.format('{}_{}'.format(sta_folder.name,sac_file.name)))
                    continue
                try:
                    st = obspy.read(str(sac_file))
                except:
                    continue
                sta_dict = dict()
                print('On {}_{}'.format(sta_folder.name , sac_file.name))
                print('Done Reading...')
                st.merge(fill_value='interpolate')

                try:
                    st.trim(starttime=st[0].stats.starttime, endtime=st[0].stats.endtime, pad=True, fill_value=0)
                except:
                    print('Trim Error')
                    continue
                for trace in st.traces:
                    net = trace.stats.network
                    sta = trace.stats.station
                    net_sta_key = net + '.' + sta
                    if net_sta_key in sta_dict.keys():
                        pass
                    else:
                        sta_dict[net_sta_key] = list()
                sta_dict_key_list = list(sta_dict.keys())
                t_cur_time = time.time()
                if parallel:
                    parallel_param_list = list()
                    parallel_gap = int(np.ceil(float(len(sta_dict_key_list))/parallel_num))
                    thread_list = list()
                    for parallel_dx in range(parallel_num):
                        parallel_param_dict = dict()
                        # Load models
                        parallel_param_dict['st'] = st
                        parallel_param_dict['sta_dict'] = sta_dict
                        parallel_param_dict['fname'] = '{}_{}'.format(sta_folder.name , sac_file.name)
                        parallel_param_dict['plot_folder'] = plot_folder 
                        parallel_param_dict['waveform_folder'] = waveform_folder 
                        parallel_param_dict['csv_folder'] = csv_folder
                        parallel_param_dict['GPU_ID'] = cfgs['GPU_ID']
                        parallel_param_dict['picker_model_path'] = cfgs['picker_model_path']
                        parallel_param_dict['motion_model_path'] = cfgs['motion_model_path']
                        parallel_param_dict['max_psdiff'] = cfgs['max_psdiff']
                        parallel_param_dict['min_psdiff'] = cfgs['min_psdiff']
                        if parallel_gap*(parallel_dx+1) > len(sta_dict_key_list):
                            parallel_param_dict['sta_dict_key_list'] = sta_dict_key_list[parallel_gap*(parallel_dx):]
                        else:
                            parallel_param_dict['sta_dict_key_list'] = sta_dict_key_list[parallel_gap*(parallel_dx):parallel_gap*(parallel_dx+1)]
                        parallel_param_list.append(parallel_param_dict)
                
                    sta_dict_joblib_res =  Parallel(n_jobs=parallel_num)(delayed(process_sta_list)(parallel_param_dict) for parallel_param_dict in parallel_param_list)
                    for dict_res in sta_dict_joblib_res:
                        for key in dict_res.keys():
                            if len(dict_res[key]) > 0:
                                if key not in sta_dict.keys():
                                    sta_dict[key] = list()
                                else:
                                    for pick_res in dict_res[key]:
                                        sta_dict[key].append(pick_res)
                else:
                    input_dict = dict()
                    input_dict['st'] = st
                    input_dict['sta_dict'] = sta_dict
                    input_dict['fname'] = '{}_{}'.format(sta_folder.name , sac_file.name)
                    input_dict['plot_folder'] = plot_folder 
                    input_dict['waveform_folder'] = waveform_folder 
                    input_dict['csv_folder'] = csv_folder
                    input_dict['GPU_ID'] = cfgs['GPU_ID']
                    input_dict['picker_model_path'] = cfgs['picker_model_path']
                    input_dict['motion_model_path'] = cfgs['motion_model_path']
                    input_dict['max_psdiff'] = cfgs['max_psdiff']
                    input_dict['min_psdiff'] = cfgs['min_psdiff']
                    input_dict['sta_dict_key_list'] = sta_dict_key_list
                    sta_dict = process_sta_list_passing_model(input_dict, picker_model, motion_model, mag_model)
                if os.path.exists(json_folder):
                    pass
                else:
                    os.makedirs(json_folder)
                # convert output format
                fname = '{}_{}'.format(sta_folder.name , sac_file.name)
                # append_stadict_to_picks_csv('csv_folder', sta_dict, fname)
                fp = open(json_folder + fname + '.json', "w")
                json.dump(sta_dict, fp)
                fp.close()
                print('One file time: {:.4f} sec'.format( time.time() - t_cur_time))