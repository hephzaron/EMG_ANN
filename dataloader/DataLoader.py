#!/usr/bin/env python
"""
@author: Daramola Tobi <hephzaron@gmail.com>
"""
import os
import sys
import argparse
import numpy as np
import scipy
import copy
import pandas as pd
from scipy.interpolate import griddata

class DataLoader(object):
    
    """DataLoader object loads dataset from root folder and give in aoutput in chunks
    Attributes:
        data_folder (str): Folder directory
        training_set (bool): Specifies the category of data to load for training and testing
        resample_fs (int): Resampling frequency of time series data set
        interpolation (str): Type of interpolation method to be used by griddata module {‘linear’, ‘nearest’, ‘cubic’},
    """
    def __init__(self, data_folder, training_set=False,resample_fs=None,  interpolation='cubic'):
        self.heading = []
        self.datasets = None
        self._chunked_data = []
        
        self._i_pos = 0
        self._j_pos = 0
        self._class_data = []
        self._resample_fs = resample_fs
        self._interpolation = interpolation
        self._data_folder = data_folder
        self._training_set = training_set
        self._df = []
        
    def __get_paths__(self, extension='.txt'):
        """Get file paths from datasets
        Args:
            extension: File extension set to .txt by default
            training_sets: Boolean value to specify set of data to import
        Returns:
            files_dir: List containing file directory
        """
        folders = sorted(os.listdir(self._data_folder))[:-1]
        files_dir = []

        if self._training_set:
            f_prefix = '2_raw_data'
        else:
            f_prefix = '1_raw_data'

        for folder in folders:
            f_path = self._data_folder+folder
            filenames = os.listdir(f_path)
            files_dir.extend(f_path +'/'+ f for f in filenames
                             if f.startswith(f_prefix) and f.endswith(extension))
        return files_dir
        
        
    def __load__(self):
        """Load datasets from folder path
        Args:
            self: Class object
        Returns:
            None
        """
        files_dir = self.__get_paths__()
        for idx, fname in enumerate(files_dir):
            if idx == 33: continue
            x = np.loadtxt(fname, skiprows=1)
            if idx == 0:
                self.heading = np.genfromtxt(fname, delimiter="\t", dtype="|U").reshape((-1,))[0:x.shape[1]]
            else: pass
            self._df.append(x)
        self.datasets = np.array(self._df)
        pass
    
    def __get_subsamples__(self):
        """Get subsamples within dataset
        Args:
            self: Class object
        Returns:
            None
        """        
        self.__load__()
        for idx in np.arange(0,len(self.datasets)):
            df = pd.DataFrame(data=self.datasets[idx],columns=self.heading, index=None)
            df = df[df['class'] !=0 ]
            
            chunk_n_x_label_1 = np.array([])
            chunk_n_x_label_2 = np.array([])
            chunk_size_per_label = []
            
            for label in df['class'].unique().tolist(): 
                #get the time difference between each timestamp 
                time_data = df['time'][df['class']==label]
                time_diffs = pd.Series(time_data).diff(periods=1)
                leap_point = np.where(time_diffs >100)
                pos = leap_point[0].item()
                #print('label-{}, position-{}'.format(label, pos))
                    
                chunk1 = df[df['class']==label].iloc[0:pos,:]
                chunk2 = df[df['class']==label].iloc[pos:,:]
                #print(chunk1)
                #print('label-{}, len-{}'.format(label, (len(chunk1), len(chunk2))))
                
                time1 = np.array(time_data)[0:pos].reshape((-1,1))
                time2 = np.array(time_data)[pos:].reshape((-1,1))
                
                time_series1 = np.concatenate((time1, np.array(chunk1)[:,-9:]), axis=1)
                time_series2 = np.concatenate((time2, np.array(chunk2)[:,-9:]), axis=1)
                
                chunk_n_x_label_1_1 = np.concatenate((chunk_n_x_label_1.reshape(-1,10), time_series1), axis=0)
                chunk_n_x_label_2_2 = np.concatenate((chunk_n_x_label_2.reshape(-1,10), time_series2), axis=0)
                
                chunk_n_x_label_1 = chunk_n_x_label_1_1
                chunk_n_x_label_2 = chunk_n_x_label_2_2
                chunk_size_per_label.append(np.array([len(chunk1), len(chunk2)]))
            self._class_data.append(np.array(chunk_size_per_label))
            self._chunked_data.append(np.array([chunk_n_x_label_1, chunk_n_x_label_2]))                           
            pass
        
    def __resample__(self, data, resample_fs):
        """Resample data
        Args:
            data (np.array) : data to be resampled
            resample_fs (int) : resampling frequency
        Returns:
            newsignal (tuple) : Tuple contains resampled data and labels
        """
        # need to round in case it's not exact
        N = len(data)
        timevec = np.linspace(0,3,N)
        newNpnts = np.round(N* (resample_fs/np.round(N/3)))
        newsignal = pd.DataFrame(index=range(int(newNpnts)))
        
        # new time vector after upsampling
        timevec_new = np.arange(0,newNpnts) / resample_fs
        df = pd.DataFrame(data, columns=self.heading, index=None)
        
        for idx, column in enumerate(df.columns):    
            # interpolate using griddata
            newsignal[column] = griddata(timevec, df[column], timevec_new, method=self._interpolation)
        newsignal['time'] = timevec_new
        return (np.array(newsignal)[:,0:9], np.array(newsignal)[:,-1])
        
    def __groupby_label__(self, data_in):
        """Load datasets from folder path
        Args:
            data_in(np.array): Time series data for all labels in a chunk
        Returns:
            data_out(np.array): Resampled output data
            label_out(np.array): Resampled output label
        """
        
        N = len(data_in)
        df = pd.DataFrame(data_in, columns=self.heading, index=None)
        grouped_data = df.groupby("class")
        
        data_out = np.array([])
        label_out = np.array([])
        
        # iterate over grouped data by class: key
        for key, batch in grouped_data:
            data, label = self.__resample__(np.array(batch), self._resample_fs)
            # Fill sampled class with actual labels
            label.fill(key)
            data_x1 = np.concatenate((data_out.reshape(-1,9), data), axis=0)
            label_x1 = np.concatenate((label_out.reshape(-1,1), label.reshape(-1,1)), axis=0)
            
            data_out = data_x1
            label_out = label_x1
        return (data_out, label_out)
        
    def __count__(self):
        """Monitors data indexing when fetching chunks of data
        Args:
            None
        Returns:
            None
        """
        self._j_pos +=1
        if (self._j_pos >= 2):
            self._j_pos = 0
            self._i_pos +=1
        else: pass
        
    def __get_chunk__(self, data_in):
        """Generator function to load chuinks of data
        Args:
            data_in (np.array) : data to be loaded chunk-wise
        Returns:
            d_out (iterator): iterator object of chunked data
        """
        stop = len(data_in)
        while self._i_pos < stop:
            
            batch = data_in[self._i_pos][self._j_pos]
            data = batch[:,0:9]
            label = batch[:,-1]
            d_out = (data, label)
            
            if (self._resample_fs):
                data, label = self.__groupby_label__(batch)
                d_out = (data, label)
                yield d_out
            else:
                yield d_out
            self.__count__()
            
    def get_data(self):
        """Fetches chunked data
        Args:
            None
        Returns:
            data_out (np.array) : chunked data
        """
        self.__get_subsamples__()
        data_out = self.__get_chunk__(self._chunked_data)
        return data_out

def get_args():
    parser = argparse.ArgumentParser(
        """Loading Datasets""")
    parser.add_argument("-d", "--datapath", type=str,  default="/",
                        help="Directory of folder containing dataset")
    parser.add_argument("-t", "--type", type=bool, default=True,
                        help="Load training or test data, default: training set")
    parser.add_argument("-s", "--newsrate", type=int, default=None,
                        help="New sampling rate for resampling data")
    parser.add_argument("-i", "--interpolation", type=str, choices=["linear", "nearest", "cubic"] ,default = "cubic",
                        help="Interpolation metod to use for resampling")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    dataloader = DataLoader(
        data_folder=opt.datapath,
        training_set=opt.type,
        resample_fs=opt.newsrate,
        interpolation=opt.interpolation)
    dataloader.get_data()