import csv
import glob
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from collections import defaultdict
from pyteomics import  fasta, parser, mass, achrom, electrochem, auxiliary, mzxml

def get_tsv_data(path):
    with open(path) as tsv:
        reader = csv.reader(tsv, dialect="excel-tab")
        header = next(reader) #skip header
        for line in reader:
            yield list(zip(header, line))


            
def get_mzxml_filenames(dir_path):
    for mzxml_filepath in glob.glob(f"{dir_path}/*.mzXML"):
        yield mzxml_filepath

class Data:
    def __init__(self, patient, pepti, intensity):
        '''
        patient is a list of patient name corresponding to x-asix in intensity
        pepti is a list of pepti id corresponsing to y-axis in intensity
        intensity is a 2D numpy array what stores intensity data of patient and pepti
        '''
        self.patient = patient
        self.pepti = pepti
        self.intensity = intensity 

    @staticmethod
    def intensity_string_to_int(intensity):
        try:
            return int(intensity.replace(',', ''))
        except ValueError:
            return -1

    @staticmethod
    def IsPatientKey( key ):
        if key.startswith("Patient_") and not key.endswith("_unmod"):
            return True
        return False

    @classmethod
    def fromFilePath(cls, path, pepti_id_col = 1, start_col = 32):
        #count line of file
        num_pepti = 0
        for _ in open(path):
            num_pepti += 1
        num_pepti -= 1 # remove header

        #open file
        with open(path) as tsv:
            reader = csv.reader(tsv, dialect="excel-tab")

            header = np.array(next(reader))
            header_mask = [cls.IsPatientKey(k) for k in header]
            patient = list(header[ header_mask ]) #header after 32 is patient name with time stamp

            pepti = [""] * num_pepti
            intensity_data = np.empty((num_pepti, len(patient)), int)

            for i, line in enumerate(reader):
                pepti[i] = line[ pepti_id_col ]
                intensity_data[i,:] = np.array([0]*start_col + [ cls.intensity_string_to_int(l) for l in line[start_col:]])[ header_mask ]

        return cls(patient, pepti, intensity_data)

    def get_patient_from_index(self,p_index):
        return self.intensity[:,p_index]

    def get_patient(self, patient):
        p_index = self.patient.index(patient)
        return self.get_patient_from_index(p_index)

    def get_patient_list(self, patient_list):
        re = np.zeros((len(self.pepti), len(patient_list)))
        for i, pat in enumerate(patient_list):
            re[:,i] = self.get_patient(pat)
        return re

    def get_pepti_from_index(self,p_index):
        return self.intensity[p_index,:]

    def get_pepti(self, pepti):
        p_index = self.pepti.index(pepti)
        return self.get_pepti_from_index(p_index)

    def get_pepti_list(self, pepti_list):
        re = np.zeros((len(pepti_list), len(self.patient)),int)
        for i, pep in enumerate(pepti_list):
            re[i,:] = self.get_pepti(pep)
        return re

if __name__ == "__main__":
    #loading data
    data = Data.fromFilePath("./data/data.tsv")

    #showing what data contains
    print(data.intensity.shape)

    print(len(data.patient))
    print(data.patient[:10])

    print(len(data.pepti))
    print(data.pepti[:10])