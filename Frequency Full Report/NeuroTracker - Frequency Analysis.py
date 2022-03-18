import numpy as np
import pandas as pd
import seaborn as sns
import glob
import pickle
import copy
import time

import mne
from mne import Epochs, pick_channels, find_events
from mne.io import read_raw_bdf, concatenate_raws
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

import collections
from autoreject import AutoReject

import sys
sys.path.append('/Users/nick/Documents/PhD/CLCT/NeuroTracker/')
from cda_utils import get_epochs, get_CDA, prep_report, fill_report, get_report, checksum, add_cda_report, get_CDA_perf_report
#from events import get_specific_events

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

mne.viz.set_3d_backend('pyvistaqt')

#==================================================================
# General Variables
#==================================================================
# Organize Triggers
trigger = 2

# Triggers
# 1 - New Spheres.
# 2 - Red Spheres.
# 4 - Start Moving.
# 8 - Stop Moving.
# 16 - TBD (~4 of them... and 4 blocks.)
# 32 - TBD (juste juste avant le 1, so I guess fin de trial, I have 88 of them... 4 trials)
# 

# T2 = T1 + 5s 
# T4 = T2 + 2s
# T8 = T4 + 9s ?! (9s ?)
# T16 = T8 + ~1s
# T32 = T8 + 25s   ensuite un autre 3s plus tard.

# Beaucoup de 1 et 32.. pas clair...

# --------------------------
conds = ['1', '2', '3']
sides = ['right', 'left', 'classicmod']
perfs = ['good', 'bad']

custom_trigger = dict()
custom_trigger['1'] = 100
custom_trigger['2'] = 200
custom_trigger['3'] = 300
custom_trigger['right'] = 10
custom_trigger['left'] = 20
custom_trigger['classicmod'] = 30
custom_trigger['good'] = 1
custom_trigger['bad'] = 2

#report = prep_report(conds, sides, perfs)

# Making Event ID Dict to make it easy and readable to access (the desired) epochs.
event_id = dict()
for cond in conds:
    for side in sides:
        for perf in perfs:
            event_id['{}/{}/{}'.format(cond, side, perf)] = custom_trigger[cond] + custom_trigger[side] + custom_trigger[perf]

# CDA Channels
chan_right_all = ['O2', 'PO4', 'PO8', 'P2', 'P4', 'P6', 'P8', 'P10', 'C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6', 'T8', 'TP8', 'FT8', 'FC2', 'FC4', 'FC6', 'F2', 'F4', 'F6', 'F8', 'AF4', 'AF8', 'Fp2']
chan_left_all =  ['O1', 'PO3', 'PO7', 'P1', 'P3', 'P5', 'P7', 'P9',  'C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5', 'T7', 'TP7', 'FT7', 'FC1', 'FC3', 'FC5', 'F1', 'F3', 'F5', 'F7', 'AF3', 'AF7', 'Fp1']
chan_midline_all = ['Fz', 'FCz', 'Cz', 'Fpz', 'CPz', 'Pz', 'POz', 'Oz'] #'Iz'

clusters = ['F', 'P', 'C', 'O', 'T']
clusters_names = {'F':'Frontal', 'P':'Parietal', 'C':'Central', 'O':'Occipital', 'T':'Temporal'}

phases = ['ID', 'Tracking', 'Recall']
stimx = {'ID':0, 'Tracking':2, 'Recall':11}
time_windows = {'ID':(-.2,2), 'Tracking':(1.5, 11), 'Recall':(10.5, 18)}

# Epoch Information
epoch_length = 20
epoch_tmin = -3
baseline_corr = (-1, -0.2)

def process_participant(participant, raw, events, save=True):
    # -- Prep MNE-Report --
    report_subj = mne.Report(title='Subject Report Raw Frequency - {}'.format(participant))
    report_subj_filename = 'Subject_Report-{}.html'.format(participant[:-4])

    # Remove Events ID if not in file.
    print('==================================================')
    print('REMOVING EVENTS ID NOT IN EEG FILE...')
    cur_event_id = copy.deepcopy(event_id)
    newevents_in_file = set(np.array(events)[:,2])
    keys_to_delete = []
    for e in cur_event_id.keys():
        if event_id[e] not in newevents_in_file:
            print('Event: {} ({}), not present.'.format(e, event_id[e]))
            keys_to_delete.append(e)
    for key in keys_to_delete:
        del cur_event_id[key]
    print('==================================================') 
    
    # Calculate average response time.
    resp_time_in_ticks = []
    for i, e in enumerate(events):
        if e[2] == 8:
            if events[i+1,2] == 32:
                resp_time_in_ticks.append(events[i+1,0] - events[i,0])
            elif events[i+1,2] == 16: #Synch?!
                if events[i+2,2] == 32:
                    resp_time_in_ticks.append(events[i+2,0] - events[i,0])
                else:
                    print('No Response?! Trigger: {}'.format(events[i+1,2]))
            else:
                print('No Response?! Trigger: {}'.format(events[i+1,2]))

    mean_rt = np.mean(resp_time_in_ticks)
    median_rt = np.median(resp_time_in_ticks)
    std_rt = np.std(resp_time_in_ticks)
    print('Response Time ~{:.2f}s (std:{:.2f})'.format(mean_rt/sfreq, std_rt/sfreq))
        
    # Epoching
    # TODO: take values from const.
    epochs = Epochs(raw, events, event_id=cur_event_id, tmin=-3, tmax=20, baseline=(None, None), preload=True, event_repeated='merge', on_missing='warn')
    
    # Cleaning
    # TODO: Generalize the n_jobs as const.
    ar = AutoReject(n_jobs=10)
    epochs, reject_log = ar.fit_transform(epochs, return_log=True) 
    fig, ax = plt.subplots(figsize=(15,8))
    reject_log.plot('horizontal', ax=ax, show=False)

    print('[Raw] Adding figure to Report...')
    report_subj.add_raw(raw=raw.pick_types(eeg=True), butterfly=False, title='Raw', tags=('General',))
    
    print('[Cleaning] Adding figure to Report...')
    report_subj.add_figure(fig=fig, title='Reject Log', caption='Reject Log', tags=('General',), image_format='PNG')

    print('[Epochs] Adding figure to Report...')
    report_subj.add_epochs(epochs=epochs, title='Epochs (Clean)', tags=('General',))
    
    # Adding Response Time to Report
    substr = '<p> Response Time: Mean= {:.2f}s | Median= {:.2f} | Std= {:.2f} </p>'.format(mean_rt/sfreq, std_rt/sfreq, median_rt/sfreq)
    report_subj.add_html(html=substr, title='Response Time', tags=('General',))
        
    # PSD
    report_subject_psd(report_subj, epochs)
    report_subject_psd_color(report_subj, epochs)

    # ERSP
    report_subject_ersp(report_subj, epochs)
    report_subject_ersp_clusters(report_subj, epochs)
    
    # TOPO Maps
    report_subject_freqs_topo(report_subj, epochs)

    print('Saving report...')
    report_subj.save(report_subj_filename, overwrite=True, open_browser=False)
    
    if save:
        # Save Particpant's Data
        data_pickle = dict()
        data_pickle['raw'] = raw
        data_pickle['events'] = events
        data_pickle['epochs'] = epochs
        with open(f"Subject_{participant[:-4]}_cleanAR.pkl","wb") as f:
            pickle.dump(data_pickle, f)


def report_subject_psd(report, epochs):
    print('[PSD] Starting...')
        
    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    # 1. Full Sequence
    psds, freqs = psd_multitaper(epochs['good'].copy().crop(tmin=0, tmax=15), fmin=2, fmax=100, n_jobs=8)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    axs[0,0].plot(freqs, psds_mean, color='k')
    axs[0,0].fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std, color='k', alpha=.5)
    axs[0,0].set(title='Multitaper PSD - Full Sequence (Good)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')

    # 2. ID (Cue)
    psds, freqs = psd_multitaper(epochs['good'].copy().crop(tmin=0, tmax=2), fmin=2, fmax=100, n_jobs=8)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    axs[0,1].plot(freqs, psds_mean, color='k')
    axs[0,1].fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std, color='k', alpha=.5)
    axs[0,1].set(title='Multitaper PSD - ID (Good)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')

    # 3. Tracking
    psds, freqs = psd_multitaper(epochs['good'].copy().crop(tmin=2, tmax=10.5), fmin=2, fmax=100, n_jobs=8)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    axs[1,0].plot(freqs, psds_mean, color='k')
    axs[1,0].fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std, color='k', alpha=.5)
    axs[1,0].set(title='Multitaper PSD - Tracking (Good)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')

    # 4. Recall
    psds, freqs = psd_multitaper(epochs['good'].copy().crop(tmin=10.5, tmax=15), fmin=2, fmax=100, n_jobs=8)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    axs[1,1].plot(freqs, psds_mean, color='k')
    axs[1,1].fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std, color='k', alpha=.5)
    axs[1,1].set(title='Multitaper PSD - Recall (Good)', xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')

    fig.tight_layout()
    #plt.show()

    print('[PSD] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='PSD',#.format(phase, perf.capitalize()),
    caption='PSD', #- L vs R ({})'.format(phase, perf.capitalize()),
    tags=('PSD',),
    image_format='PNG')
    
    print('[PSD] Done!')
    

def report_subject_psd_color(report, epochs):   
    print('[PSD] Starting...')
    
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    
    epochs['good'].copy().crop(tmin=0, tmax=15).plot_psd(fmin=1, fmax=50, ax=axs[0,0], show=False)
    epochs['good'].copy().crop(tmin=0, tmax=2).plot_psd(fmin=1, fmax=50, ax=axs[0,1], show=False)
    epochs['good'].copy().crop(tmin=2, tmax=10.5).plot_psd(fmin=1, fmax=50, ax=axs[1,0], show=False)
    epochs['good'].copy().crop(tmin=10.5, tmax=17).plot_psd(fmin=1, fmax=50, ax=axs[1,1], show=False)
    axs[0,0].set_title('PSD - Full Sequence')
    axs[0,1].set_title('PSD - ID')
    axs[1,0].set_title('PSD - Tracking')
    axs[1,1].set_title('PSD - Recall')
    
    fig.tight_layout()
    #plt.show()

    print('[PSD] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='PSD',#.format(phase, perf.capitalize()),
    caption='PSD', #- L vs R ({})'.format(phase, perf.capitalize()),
    tags=('PSD',),
    image_format='PNG')
    
    print('[PSD] Done!')


FREQ_BANDS = dict()
FREQ_BANDS['Full Spectrum'] = (1, 100)
FREQ_BANDS['Delta'] = (1, 4)
FREQ_BANDS['Theta'] = (4, 8)
FREQ_BANDS['Alpha'] = (8, 12)
FREQ_BANDS['Beta'] = (12, 35)
FREQ_BANDS['Gamma'] = (35, 100)

TICK_DIV = {'Delta':1, 'Theta':1, 'Alpha':1, 'Beta':5, 'Gamma':10, 'Full Spectrum':10 }

def report_subject_ersp(report, epochs, baseline=(-1,0), mode='logratio'):
    print('[ERSP] Starting...')
    
    freqs = np.arange(1, 100, 1)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3, n_jobs=8)
    #power_data_db = 10*np.log10(power.data) # convert to dB
    power = power.apply_baseline(baseline, mode=mode)
    power_data_db = power.data
    tf_times = power.times

    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    # Full Sequence
    freq_anal = power_data_db[:,:, (power.times > -1) & (power.times < 16)].mean(0)
    ticks_offset = np.where(power.times > -1)[0][0]
    times_of_interest = [0, 2, 3, 11, 15]
    ticks_of_interest = [(np.where(power.times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    #sns.heatmap(ax=axs[0,0], data=freq_anal)#, yticklabels=freqs)
    z = axs[0,0].imshow(freq_anal[:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[0,0].set_xticks(ticks_of_interest)
    axs[0,0].set_xticklabels(times_of_interest)
    axs[0,0].invert_yaxis()
    axs[0,0].set_title('Full Sequence')
    plt.colorbar(z, ax=axs[0,0])

    # ID
    freq_anal = power_data_db[:,:, (power.times > -0.2) & (power.times < 2)].mean(0)
    ticks_offset = np.where(power.times > -0.2)[0][0]
    times_of_interest = [0, 1]
    ticks_of_interest = [(np.where(power.times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[0,1].imshow(freq_anal[:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[0,1].set_xticks(ticks_of_interest)
    axs[0,1].set_xticklabels(times_of_interest)
    axs[0,1].invert_yaxis()
    axs[0,1].set_title('ID')
    plt.colorbar(z, ax=axs[0,1])

    # Tracking
    freq_anal = power_data_db[:,:, (power.times > 1.8) & (power.times < 10.5)].mean(0)
    ticks_offset = np.where(power.times > 1.8)[0][0]
    times_of_interest = [2, 3, 6, 10]
    ticks_of_interest = [(np.where(power.times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[1,0].imshow(freq_anal[:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[1,0].set_xticks(ticks_of_interest)
    axs[1,0].set_xticklabels(times_of_interest)
    axs[1,0].invert_yaxis()
    axs[1,0].set_title('Tracking')
    plt.colorbar(z, ax=axs[1,0])

    # Recall
    freq_anal = power_data_db[:,:, (power.times > 10.5) & (power.times < 16)].mean(0)
    ticks_offset = np.where(power.times > 10.5)[0][0]
    times_of_interest = [11, 12, 13, 14, 15]
    ticks_of_interest = [(np.where(power.times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[1,1].imshow(freq_anal[:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[1,1].set_xticks(ticks_of_interest)
    axs[1,1].set_xticklabels(times_of_interest)
    axs[1,1].invert_yaxis()
    axs[1,1].set_title('Recall')
    plt.colorbar(z, ax=axs[1,1])

    fig.tight_layout()
    #plt.show()

    print('[ERSP] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='ERSP',#.format(phase, perf.capitalize()),
    caption='ERSP', #- L vs R ({})'.format(phase, perf.capitalize()),
    tags=('ERSP',),
    image_format='PNG')

    
    # -------- ITC --------
    fig = itc.plot_topo(title='Inter-Trial coherence', cmap='Reds', show=False)
    
    print('[ERSP/ITC] Adding figure to Report...')
    #report.add_figure(
    #fig=fig, title='ITC',#.format(phase, perf.capitalize()),
    #caption='ITC', #- L vs R ({})'.format(phase, perf.capitalize()),
    #tags=('Electrodes','ITC',),
    #image_format='PNG')
    itc_filename = str(report.title[report.title.rfind('-') + 2:-4]) + str('_itc.png')
    fig.savefig(itc_filename, format='png', dpi=300)
    report.add_html(html='<img src="{}">'.format(itc_filename), title='ITC', tags=('ITC',))
    
    print('[ERSP] Done!')

        
def report_subject_ersp_clusters(report, epochs, baseline=(-1,0), mode='logratio'):
    print('[ERSP] (clusters) Starting...')
    
    print('[ERSP] (clusters) Calculating Power.')
    freqs = np.arange(1, 100, 1)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3, n_jobs=8)
    tf_times = power.times
    #power_data_db = 10*np.log10(power.data) if dB else power.data
    power = power.apply_baseline(baseline, mode=mode)
    print('[ERSP] (clusters) Done Calculating Power!')
    
    # Finding Channels per Cluster
    ch_per_cluster = dict()
    for cl in clusters:
        ch_per_cluster[cl] = []
        for ch in power.info.ch_names:
            if ch.rfind(cl) >= 0:
                ch_per_cluster[cl].append(ch)
    
    for f_band in FREQ_BANDS.keys():
        print('[ERSP] Doing {} Band'.format(f_band))
        cur_power = power.copy().crop(fmin=FREQ_BANDS[f_band][0], fmax=FREQ_BANDS[f_band][1])
        cur_power_data = cur_power.data
        cur_y_values = [(i, f) for i, f in enumerate(cur_power.freqs) if f % TICK_DIV[f_band] == 0]
        cur_freqs_ticks = [val[0] for val in cur_y_values]
        cur_freqs_labels = [val[1] for val in cur_y_values]
        
        freq_anal_cl = dict()
        for cl in clusters:
            print('[ERSP] Doing Cluster: {} ({})'.format(clusters_names[cl], ch_per_cluster[cl]))
            cl_start_time = time.time()

            fig, axs = plt.subplots(len(ch_per_cluster[cl]), 4, figsize=(16, len(ch_per_cluster[cl]) * 4))

            cur_freq = dict()
            freq_anal_cl[cl] = dict()
            freq_full = np.array([])
            freq_id = np.array([])
            freq_tracking = np.array([])
            freq_recall = np.array([])

            for i, ch in enumerate(ch_per_cluster[cl]):
                cur_freq["Full"] = cur_power_data[i, :, (power.times > -1) & (power.times < 16)].transpose()
                cur_freq["ID"] = cur_power_data[i, :, (power.times > -0.2) & (power.times < 2)].transpose()
                cur_freq["Tracking"] = cur_power_data[i, :, (power.times > 1.8) & (power.times < 10.5)].transpose()
                cur_freq["Recall"] = cur_power_data[i, :, (power.times > 10.5) & (power.times < 16)].transpose()

                plot_ersp_row(cur_freq, power.times, cur_freqs_ticks, cur_freqs_labels, ch, axs[i,:])

                freq_full =  np.dstack((freq_full, cur_freq["Full"])) if len(freq_full) else cur_freq["Full"]
                freq_id = np.dstack((freq_id, cur_freq["ID"])) if len(freq_id) else cur_freq["ID"]
                freq_tracking = np.dstack((freq_tracking, cur_freq["Tracking"])) if len(freq_tracking) else cur_freq["Tracking"]
                freq_recall = np.dstack((freq_recall, cur_freq["Recall"])) if len(freq_recall) else cur_freq["Recall"]

            fig.tight_layout()
            #plt.show()

            print('Adding figure to Report...')
            report.add_figure(
            fig=fig, title='ERSP - {}/{} (Chs)'.format(f_band, clusters_names[cl]),#.format(phase, perf.capitalize()),
            caption='ERSP - {}/{} (Chs)'.format(f_band, clusters_names[cl]), #- L vs R ({})'.format(phase, perf.capitalize()),
            tags=('Electrodes',),
            image_format='PNG')

            # Getting the Cluster Average.
            freq_anal_cl[cl]["Full"] = freq_full.mean(2)
            freq_anal_cl[cl]["ID"] = freq_id.mean(2)
            freq_anal_cl[cl]["Tracking"] = freq_tracking.mean(2)
            freq_anal_cl[cl]["Recall"] = freq_recall.mean(2)

            print('[ERSP] Cluster: {}, done! ({})'.format(clusters_names[cl], time.time() - cl_start_time))

        # Making an All Clusters Figure. (easier to look at and compare clusters)
        fig, axs = plt.subplots(len(clusters), 4, figsize=(16, len(clusters) * 4))
        for i, cl in enumerate(clusters):
            plot_ersp_row(freq_anal_cl[cl], power.times, cur_freqs_ticks, cur_freqs_labels, clusters_names[cl], axs[i, :])

        fig.tight_layout()
        #plt.show()

        print('Adding figure to Report...')
        report.add_figure(
        fig=fig, title='ERSP - {}/All Clusters'.format(f_band),
        caption='ERSP - {}/All Clusters'.format(f_band),
        tags=('Electrodes','ERSP','Clusters'),
        image_format='PNG')
        
        print('[ERSP] {} Band Done!'.format(f_band))

    print('[ERSP] (clusters) Done!')
        

def plot_ersp_row(freq_anal, times, yticks, yticklabels, ch, axs):
    # Full Sequence
    ticks_offset = np.where(times > -1)[0][0]
    times_of_interest = [0, 2, 3, 11, 15]
    ticks_of_interest = [(np.where(times >= t)[0][0] - ticks_offset) for t in times_of_interest]    
    z = axs[0].imshow(freq_anal["Full"][:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[0].set_xticks(ticks_of_interest)
    axs[0].set_xticklabels(times_of_interest)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(yticklabels)
    axs[0].invert_yaxis()
    axs[0].set_title(f'[{ch}] - Full Sequence')
    axs[0].set_ylabel(ch)
    plt.colorbar(z, ax=axs[0])

    # ID
    ticks_offset = np.where(times > -0.2)[0][0]
    times_of_interest = [0, 1]
    ticks_of_interest = [(np.where(times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[1].imshow(freq_anal["ID"][:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[1].set_xticks(ticks_of_interest)
    axs[1].set_xticklabels(times_of_interest)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    axs[1].invert_yaxis()
    axs[1].set_title(f'[{ch}] - ID') 
    plt.colorbar(z, ax=axs[1])

    # Tracking
    ticks_offset = np.where(times > 1.8)[0][0]
    times_of_interest = [2, 3, 6, 10]
    ticks_of_interest = [(np.where(times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[2].imshow(freq_anal["Tracking"][:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[2].set_xticks(ticks_of_interest)
    axs[2].set_xticklabels(times_of_interest)
    axs[2].set_yticks(yticks)
    axs[2].set_yticklabels(yticklabels)
    axs[2].invert_yaxis()
    axs[2].set_title(f'[{ch}] - Tracking')
    plt.colorbar(z, ax=axs[2])

    # Recall
    ticks_offset = np.where(times > 10.5)[0][0]
    times_of_interest = [11, 12, 13, 14, 15]
    ticks_of_interest = [(np.where(times >= t)[0][0] - ticks_offset) for t in times_of_interest]
    z = axs[3].imshow(freq_anal["Recall"][:,100:-100], cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axs[3].set_xticks(ticks_of_interest)
    axs[3].set_xticklabels(times_of_interest)
    axs[3].set_yticks(yticks)
    axs[3].set_yticklabels(yticklabels)
    axs[3].invert_yaxis()
    axs[3].set_title(f'[{ch}] - Recall')
    plt.colorbar(z, ax=axs[3])



def report_subject_freqs_topo(report, epochs, dB=True): 
    #---------------------------------------------------------
    # Plot Topo for Good Trials - 3 Phases.
    #---------------------------------------------------------
    fig, axs = plt.subplots(3, 5, figsize=(16, 12))
    epochs['good'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[0,:], show=False, n_jobs=5)
    epochs['good'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[1,:], show=False, n_jobs=5)
    epochs['good'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[2,:], show=False, n_jobs=5)
    axs[0,0].set_ylabel('ID (Good)')
    axs[1,0].set_ylabel('Tracking (Good)')
    axs[2,0].set_ylabel('Recall (Good)')
    fig.tight_layout()
    #plt.show()
    
    print('[TOPO] Adding figure...')
    report.add_figure(
    fig=fig, title='Topo - Good',
    caption='Topo - Good',
    tags=('Topo',),
    image_format='PNG')
    
    
    #---------------------------------------------------------
    # Plot Topo for Bad Trials - 3 Phases.
    #---------------------------------------------------------
    fig, axs = plt.subplots(3, 5, figsize=(16, 12))
    epochs['bad'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[0,:], show=False, n_jobs=5)
    epochs['bad'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[1,:], show=False, n_jobs=5)
    epochs['bad'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[2,:], show=False, n_jobs=5)
    axs[0,0].set_ylabel('ID (Bad)')
    axs[1,0].set_ylabel('Tracking (Bad)')
    axs[2,0].set_ylabel('Recall (Bad)')
    fig.tight_layout()
    #plt.show()
    
    print('[TOPO] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='Topo - Bad',
    caption='Topo - Bad',
    tags=('Topo',),
    image_format='PNG')
    
    
    #---------------------------------------------------------
    # Plot Topo for Left vs Right Trials - 3 Phases.
    #---------------------------------------------------------
    fig, axs = plt.subplots(6, 5, figsize=(18, 12))
    epochs['left'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[0,:], show=False, n_jobs=5)
    epochs['right'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[1,:], show=False, n_jobs=5)
    epochs['left'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[2,:], show=False, n_jobs=5)
    epochs['right'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[3,:], show=False, n_jobs=5)
    epochs['left'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[4,:], show=False, n_jobs=5)
    epochs['right'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[5,:], show=False, n_jobs=5)
    axs[0,0].set_ylabel('ID (Left)')
    axs[1,0].set_ylabel('ID (Right)')
    axs[2,0].set_ylabel('Tracking (Left)')
    axs[3,0].set_ylabel('Tracking (Right)')
    axs[4,0].set_ylabel('Recall (Left)')
    axs[5,0].set_ylabel('Recall (Right)')
    fig.tight_layout()
    #plt.show()
    
    print('[TOPO] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='Topo - Left/Right',
    caption='Topo - Left/Right',
    tags=('Topo',),
    image_format='PNG')
    
    
    #---------------------------------------------------------
    # Plot Topo for Set Sizes - 3 Phases.
    #---------------------------------------------------------
    fig, axs = plt.subplots(9, 5, figsize=(24, 16))
    epochs['1/good'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[0,:], show=False, n_jobs=5)
    epochs['2/good'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[1,:], show=False, n_jobs=5)
    epochs['3/good'].plot_psd_topomap(tmin=-0.2, tmax=2, ch_type='eeg', normalize=False, axes=axs[2,:], show=False, n_jobs=5)
    epochs['1/good'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[3,:], show=False, n_jobs=5)
    epochs['2/good'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[4,:], show=False, n_jobs=5)
    epochs['3/good'].plot_psd_topomap(tmin=1.8, tmax=10.5, ch_type='eeg', normalize=False, axes=axs[5,:], show=False, n_jobs=5)
    epochs['1/good'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[6,:], show=False, n_jobs=5)
    epochs['2/good'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[7,:], show=False, n_jobs=5)
    epochs['3/good'].plot_psd_topomap(tmin=10.5, tmax=15, ch_type='eeg', normalize=False, axes=axs[8,:], show=False, n_jobs=5)
    axs[0,0].set_ylabel('ID (1/Good)')
    axs[1,0].set_ylabel('ID (2/Good)')
    axs[2,0].set_ylabel('ID (3/Good)')
    axs[3,0].set_ylabel('Tracking (1/Good)')
    axs[4,0].set_ylabel('Tracking (2/Good)')
    axs[5,0].set_ylabel('Tracking (3/Good)')
    axs[6,0].set_ylabel('Recall (1/Good)')
    axs[7,0].set_ylabel('Recall (2/Good)')
    axs[8,0].set_ylabel('Recall (3/Good)')
    fig.tight_layout()
    #plt.show()
    
    print('[TOPO] Adding figure to Report...')
    report.add_figure(
    fig=fig, title='Topo - Set Sizes',
    caption='Topo - Set Sizes',
    tags=('Topo',),
    image_format='PNG')   
    
    
# Plot Joint to show Topo for 1 specific point in time to show ERP like activity.
# Would be interesting to having a small evolution in time (gif like...)
def report_subject_ersp_topo(report, epochs, dB=True):    
    # ---------------------------------
    # Full Spectrum
    # ---------------------------------
    print('TODO')
    #x = power.plot_joint(dB=True, baseline=(-1, 0), mode='mean', tmin=0, tmax=18, fmin=0, fmax=50, timefreqs=[(0.5, 10), (2.0, 10)])
    #x = power.plot_joint(dB=True, baseline=(-1, 0), mode='mean', tmin=0, tmax=2, fmin=0, fmax=50, timefreqs=[(0.5, 10), (1.0, 10)])
    #x = power.plot_joint(dB=True, baseline=(1.5, 2), mode='mean', tmin=2, tmax=10.5, fmin=0, fmax=50, timefreqs=[(2.5, 10), (8, 10)])
    #x = power.plot_joint(dB=True, baseline=(10, 10.5), mode='mean', tmin=10.5, tmax=15, fmin=0, fmax=50, timefreqs=[(12, 10)])



#==================================================================
# Main Loop - Loading All Files & Extract CDAs.
#==================================================================
directory_path = '/Users/nick/Documents/PhD/CAE - InLook/NeuroTracker - Sub TEMP 3/'

#filepath = directory_path + 'M22F2.bdf'
for filepath in glob.iglob(directory_path + '*.bdf'):
    filename = filepath[filepath.rfind('/')+1:]
    
    # Skipping Exceptions with multiple EEG files. (they will be loaded with Block 1 (B1))
    if filename in ['M8M1_B3.bdf','M8M1_B4.bdf','M8M2_B3.bdf','M8M2_B4.bdf','T9F2_B3-B4.bdf','T9M1_B3-B4.bdf']:
        continue
    
    start_time_subj = time.time()
    print('Processing: ' + filename + '...')

    # 1. -- Load File --
    if filename not in ['M8M1_B1-B2.bdf', 'M8M2_B1-B2.bdf', 'T9F2_B1-B2.bdf', 'T9M1_B1-B2.bdf']: # Check for exceptions with multiple EEG files.
        raw = read_raw_bdf(filepath, preload=True)
    else:
        if filename == 'M8M1_B1-B2.bdf':
            raw1 = read_raw_bdf(filepath, preload=True)
            raw2 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'M8M1_B3.bdf', preload=True)
            raw3 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'M8M1_B4.bdf', preload=True)
            raw = concatenate_raws([raw1, raw2, raw3])
            filename = 'M8M1.bdf'
        if filename == 'M8M2_B1-B2.bdf':
            raw1 = read_raw_bdf(filepath, preload=True)
            raw2 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'M8M2_B3.bdf', preload=True)
            raw3 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'M8M2_B4.bdf', preload=True)
            raw = concatenate_raws([raw1, raw2, raw3])
            filename = 'M8M2.bdf'
        if filename == 'T9F2_B1-B2.bdf':
            raw1 = read_raw_bdf(filepath, preload=True)
            raw2 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'T9F2_B3-B4.bdf', preload=True)
            raw = concatenate_raws([raw1, raw2])
            filename = 'T9F2.bdf'
        if filename == 'T9M1_B1-B2.bdf':
            raw1 = read_raw_bdf(filepath, preload=True)
            raw2 = read_raw_bdf(filepath[:filepath.rfind('/')+1] + 'T9M1_B3-B4.bdf', preload=True)
            raw = concatenate_raws([raw1, raw2])
            filename = 'T9M1.bdf'    

    raw = raw.drop_channels(ch_names=['EXG1','EXG2','EXG3','EXG6','EXG7', 'EXG8'])
    raw = raw.set_eeg_reference(ref_channels=['EXG4', 'EXG5'])
    raw = raw.set_channel_types({'EXG4': 'misc', 'EXG5': 'misc'})

    # Set Electrodes Locations
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage)

    # Bad channels - manual removal...
    # Subject T23F1.bdf: PO4
    # Subject T18F2.bdf: Iz
    #raw = raw.drop_channels('PO4')
    #raw = raw.drop_channels('Iz')

    # Sampling Frequency
    sfreq = raw.info['sfreq']

    # Events 
    events = find_events(raw)

    # ---------- Debug Logs ----------
    frequency = collections.Counter(events[:,2])
    events_frequency = dict(frequency)

    print('Max Events in File...')
    print("---------------------------")
    for e in set(events[:,2]):
        print("{}: \t\t{}".format(e, frequency[e]))
    print("---------------------------")
    # ---------- Debug Logs ----------

    df_behav = pd.read_csv(directory_path + filename[:filename.rfind('.')] + '.csv', sep=',')

    new_events = []
    nb_trials = 0
    total_skipped = 0
    for e in events:
        cur_cond = None
        cur_side = None
        cur_perf = None
        if e[2] == 2: # Find Start of Trial
            cur_cond = str(df_behav['NbTargets'].values[nb_trials]).lower()
            cur_side = str(df_behav['Mode'].values[nb_trials]).lower()
            cur_perf = 'good' if (int(df_behav['NbTargets'].values[nb_trials]) == int(df_behav['Results'].values[nb_trials])) else 'bad'
            if cur_cond is not None and cur_side is not None and cur_perf is not None:
                #temp_trigger = int(custom_trigger[cur_cond]) + int(custom_trigger[cur_side]) + int (custom_trigger[cur_perf])
                temp_event = [e[0], e[1], event_id['{}/{}/{}'.format(cur_cond, cur_side, cur_perf)]]
                new_events.append(temp_event) # Make a new list of events with new custom triggers.
            else:
                total_skipped = total_skipped + 1
                print('Skipping this event {}: No Cond or Side or Perf'.format(e))

            nb_trials = nb_trials + 1

            # Check Synch!
            if (nb_trials + 1) % 10 == 0:
                print('Check Synch!')
    
    print("A total of {} trials were added and {} were skipped.".format(nb_trials, total_skipped))

    process_participant(filename, raw, new_events, save=True)

    print('{} - Done (in {:.2f})!'.format(filename, time.time() - start_time_subj))