import pandas as pd 
import numpy as np
import sys

class Radar:

    def __init__(self, city, t, how='csv', path='../../data/', store='store.h5',
                 fname='Charlotte_box_radar_{yyyy}_{mm:02d}.csv',
                 llname='Charlotte_box_latlon_big.csv'):
        self.city = city.upper()
        self.t = pd.Timestamp(t)
        self.path = path
        self.llname = llname
        if how == 'csv':
            self.fname = fname.format(yyyy=self.t.year, mm=self.t.month)
            self.get_box(self.from_csv())

        if how == 'hdf5':
            self.store = store
            self.fname = '{c}_{yyyy}_{mm:02d}'.format(c=self.city, yyyy=self.t.year, mm=self.t.month)
            self.get_box(self.from_hdf5())
        print('new instance of Radar object for: {c} {yyyy}-{mm:02d}'.format(c=self.city, yyyy=self.t.year, mm=self.t.month))

    def from_csv(self):
        path = self.path+'{c}/BigBOX/'.format(c=self.city)
        def dateparse(Y, m, d, H, M):
            d = pd.datetime(int(Y), int(m), int(d), int(H), int(M))
            return d

        df = pd.read_csv(path+self.fname,
                         header=None, sep = ',', na_values = '-99',
                         parse_dates={'date_time': [0,1,2,3,4]},
                         date_parser=dateparse, index_col=[0])
        df.columns = range(0,19600)
        return df
    
    def from_hdf5(self):
        store = pd.HDFStore(self.store)
        df = store[self.fname]
        store.close()
        return df

    def get_box(self, df):
        path = self.path+'{c}/BigBOX/'.format(c=self.city)
        ll = pd.read_csv(path+self.llname, index_col=0, header=None, names=['lat','lon'])

        self.box = df.values.reshape(df.shape[0],140,140)
        self.lon = ll.lon.reshape(140,140)
        self.lat = ll.lat.reshape(140,140)
        self.time = df.index

