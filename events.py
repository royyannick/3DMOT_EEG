from operator import xor
import numpy as np
import pandas as pd
from mne import Epochs, pick_channels

#==================================================================
# General Functions
#==================================================================
def get_key(d, val):
    for k in d.keys():
        for t in d[k]:
            if t == val:
                return k

    return None


def get_specific_events(events, conds, sides, perfs, accs, triggers, internal_triggers):
    # Prep Structure
    # First trigger = Cond (which is also side), then trigger for Perf (here it's accuracy).
    triggers_cond = []
    triggers_side = []
    triggers_perf = [] 

    specific_events = dict()
    for cur_cond in conds:
        specific_events[cur_cond] = dict()
        for t in triggers[cur_cond]: triggers_cond.append(t)

        for cur_side in sides:
            specific_events[cur_cond][cur_side] = dict()
            for t in triggers[cur_side]: triggers_side.append(t)

            for cur_perf in accs:#perfs:
                specific_events[cur_cond][cur_side][cur_perf] = np.array([])
                for t in triggers[cur_perf]: triggers_perf.append(t)

    # Find Events from File.
    total_count = 0 
    total_skipped = 0
    for i, e in enumerate(events):
        if e[2] in triggers_cond: # Starting from condition. Will go forward for result and same trigger/event for side.
            cur_cond = get_key(triggers, e[2])

            # Find next perf trigger to classify this trial based on perf.
            j=i+1
            cur_perf = None
            while (cur_perf is None) and (j < len(events)):
                if events[j, 2] in triggers_perf:
                    for acc in accs:
                        if events[j, 2] in triggers[acc]:
                            cur_perf = acc                       
                else:
                    if events[j, 2] in triggers_cond:
                        #raise ValueError('Overlapping Events with no Accuracy/Perf!')
                        print('Overlapping Events with no Accuracy/Perf! Skipping...')
                        total_skipped = total_skipped + 1
                    j=j+1

            # Find previous side trigger to classify this trial based side.
            cur_side = None
            for side in sides:
                if e[2] in triggers[side]:
                    cur_side = side

            if cur_side is not None and cur_perf is not None:
                cur_event = e.copy()
                cur_event[2] = internal_triggers['{}-{}-{}'.format(cur_cond,cur_side,cur_perf)] # Modify the value to make it possible to separate them later.
                #print('{}. Adding: {} - {} - {}'.format(total_count, cur_cond, cur_side, cur_perf))
                specific_events[cur_cond][cur_side][cur_perf] = np.vstack((specific_events[cur_cond][cur_side][cur_perf], cur_event)) if len(specific_events[cur_cond][cur_side][cur_perf]) else cur_event
                total_count = total_count + 1
            else:
                print('Skipping this event {}'.format(e))
                total_skipped = total_skipped + 1
                
    for cur_cond in specific_events.keys():
        for cur_side in specific_events[cur_cond].keys():
            for cur_perf in specific_events[cur_cond][cur_side].keys():
                if (len(specific_events[cur_cond][cur_side][cur_perf].shape) == 1) and (specific_events[cur_cond][cur_side][cur_perf].shape[0] == 3):
                    specific_events[cur_cond][cur_side][cur_perf] = specific_events[cur_cond][cur_side][cur_perf].reshape((1,3))
    
    print("A total of {} events were added and {} were skipped.".format(total_count, total_skipped))

    return specific_events














    events = find_events(preproc)
    new_events = copy.deepcopy(events) # Copy of events that we'll modify to assign different triggers to use for concatening epochs
    
    frequency = collections.Counter(events[:,2])
    events_frequency = dict(frequency)
    print('Max Events in File...')
    print("---------------------------")
    for e in set(events[:,2]):
        print("{}: \t\t{}".format(e, frequency[e]))
    print("---------------------------")

    df_behav = pd.read_csv(directory_path + filename[:filename.rfind('.')] + '.csv', sep=',')
    
    specific_events = dict()
    for cond in conds:
        specific_events[cond] = dict()
        for side in sides:
            specific_events[cond][side] = dict()
            for perf in perfs:
                specific_events[cond][side][perf] = []

    nb_trials = 0
    for e in new_events:
        cur_cond = None
        cur_side = None
        cur_perf = None
        if e[2] == 2:
            cur_cond = str(df_behav['NbTargets'].values[nb_trials])
            cur_side = str(df_behav['Mode'].values[nb_trials])
            cur_perf = 'good' if (int(df_behav['NbTargets'].values[nb_trials]) == int(df_behav['Results'].values[nb_trials])) else 'bad'
            if cur_cond is not None and cur_side is not None and cur_perf is not None:
                e[2] = int(new_events_value[cur_cond]) + int(new_events_value[cur_side]) + int (new_events_value[cur_perf])
                specific_events[cur_cond][cur_side][cur_perf].append(e)

            nb_trials = nb_trials + 1
            
            # Check Synch!
            if (nb_trials + 1) % 10 == 0:
                print('Check Synch!')