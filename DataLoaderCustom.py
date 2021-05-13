import random
import torch
import numpy as np

from Data import Data
from torch.utils.data import Dataset, DataLoader


class PatientDiffLoader(Dataset):
    # UNTESTED !!!
    def __init__(self, path):
        self.data = Data.fromFilePath(path)
    def __len__(self):
        return len(self.data.patient)

    def __getitem__(self, idx, mode="same_patient", toBin = False, toLog = False):
        '''geting new data

        Args:
            idx ([type]): index for patient 1
            mode (str, optional): for future expension. Defaults to "same_patient".
            toBin (bool, optional): binarize output. Defaults to False.
            toLog (bool, optional): apply log to output. Defaults to False.

        Returns:
            data1: intensity data for patient[idx]
            data2: intensity data for a random patient at random time point
            label: is data1 the same paitent as data2 ?
        '''

        diff = bool(random.getrandbits(1))
        
        print(self.data.patient[idx])
        pid = int(self.data.patient[idx][8:10])
        tp = int(self.data.patient[idx][21])

        label = None
        data_1 = self.data.get_patient_from_index(idx)
        data_2 = None
        if diff:
            #different class
            label = 0.0

            #get a differnt pid
            diff_pid = -1
            while True:
                diff_pid = random.randint(1,58) #there is 58 patients? TODO: make sure it's 58
                if diff_pid != pid:
                    break

            #get diff data
            random_diff = f"Patient_{diff_pid:02d}.Timepoint_{random.randint(1,7)}"
            data_2 = self.data.get_patient( random_diff )
            
            
        else:
            #same class
            label = 1.0
            random_same = f"Patient_{pid:02d}.Timepoint_{random.randint(1,7)}"
            data_2 = self.data.get_patient( random_same )


        if toBin:
            data_1 = (data_1 != -1)
            data_2 = (data_2 != -1)
        elif toLog:
            data_1_mask = (data_1 == -1)
            data_2_mask = (data_2 == -1)

            new_data_1 = np.zero_like(data_1)
            new_data_2 = np.zero_like(data_2)

            new_data_1[data_1_mask] = np.log( data_1[data_1_mask] )
            new_data_2[data_2_mask] = np.log( data_2[data_2_mask] )
        else:
            data_1[data_1 < 0 ] = 0
            data_2[data_2 < 0 ] = 0

        return data_1, data_2, torch.from_numpy(np.array([label], dtype=np.float32))