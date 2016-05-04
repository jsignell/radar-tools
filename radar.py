import pandas as pd 
import numpy as np
import sys

class Radar:

    def __init__(self, city, t, how='csv', make_rate=False, path='../../data/', store='store.h5',
                 fname=None, llname='Charlotte_box_latlon_big.csv'):
        self.city = city.upper()
        self.t = pd.Timestamp(t)
        self.path = path
        self.llname = llname
        self.is_rate = False
        if how == 'csv':
            if fname is None:
                self.fname = 'Charlotte_box_radar_{yyyy}_{mm:02d}.csv'.format(yyyy=self.t.year, mm=self.t.month)
            else:
                self.fname = fname.format(c=self.city, yyyy=self.t.year, mm=self.t.month, dd=self.t.day)
            self.get_box(self.from_csv())

        if how == 'hdf5':
            self.store = store
            if fname is None:
                self.fname = '{c}_{yyyy}_{mm:02d}'.format(c=self.city, yyyy=self.t.year, mm=self.t.month)
            else:
                self.fname = fname.format(c=self.city, yyyy=self.t.year, mm=self.t.month, dd=self.t.day)
            self.get_box(self.from_hdf5())
        if make_rate:
            self.to_rate(make_rate)
        #print('new instance of Radar object for: {c} {yyyy}-{mm:02d}'.format(c=self.city, yyyy=self.t.year, mm=self.t.month))
        
    def to_rate(self, per_hour=4):
        if self.is_rate:
            print('{self} already is a rate'.format(self=self))
        else:
            #print('Multiplying by {per_hour} to get rate from accumulation'.format(per_hour=per_hour))
            self.box = self.box*per_hour
            self.is_rate = True

    def set_t(self, t):
        t = pd.Timestamp(t)
        if t in self.time:
            self.t = t
        else:
            print 'choose a time in this range {s}:{e}'.format(s=self.time[0], e=self.time[-1])
        return self.t
     
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
    
    def centralized_difference(self, t_start=None, t_end=None, radius=15, buffer=20, save=False, **kwargs):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        
        l =[]
        count=0
        if radius+3 < buffer:
            r=radius+3
        else:
            r=buffer
        ixy0 = buffer
        ixyn = self.lat.shape[1]-buffer
        try:
            it0 = self.time.get_loc(t_start)
            itn = self.time.get_loc(t_end)
        except:
            it0 = 0
            itn = self.time.shape[0]-2
        for ix in range(ixy0, ixyn):
            for iy in range(ixy0, ixyn):
                for it in range(it0, itn):
                    here = self.box[it+1, iy-r:iy+r+1, ix-r:ix+r+1]-self.box[it, iy, ix]
                    if not np.isnan(np.sum(here)): #np.isnan(here).any((0,1)): 
                        if count == 0:
                            test = here
                            count+=1
                        else:
                            test += here
                            count+=1
        test/=float(count)
        if 'vmin' not in kwargs.keys():
            peak = max(np.abs(np.min(test[r-radius:radius+r, r-radius:radius+r])), 
                       np.max(test[r-radius:radius+r, r-radius:radius+r]))
            kwargs.update(dict(vmin = -peak, vmax = peak))
        if 'nrows' in kwargs.keys():
            nrows = kwargs.pop('nrows')
            ncols = kwargs.pop('ncols')
            n = kwargs.pop('n')
        else:
            nrows = ncols = n = 1
        ax = plt.subplot(nrows, ncols, n, projection=ccrs.PlateCarree())

        scat = ax.pcolor(self.lon[iy-r:iy+r+1, ix-r:ix+r+1], self.lat[iy-r:iy+r+1, ix-r:ix+r+1], test, **kwargs)
        ax.set_extent([self.lon[iy, ix-radius], self.lon[iy, ix+radius], self.lat[iy-radius, ix], self.lat[iy+radius, ix]])
        ax.scatter(self.lon[iy, ix], self.lat[iy, ix], edgecolor='white', facecolor='None')
        return(scat, ax, kwargs['vmax'])

