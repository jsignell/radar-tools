import pandas as pd 
import numpy as np
import sys

class Radar:

    def __init__(self, city, t, how='csv', make_rate=False, 
                 csv_path='../../data/{c}/BOX/', store='{c}/store.h5',
                 fname=None, llname='{c}/box_latlon.csv'):
        self.city = city.upper()
        self.t = pd.Timestamp(t)
        self.csv_path = csv_path.format(c=self.city)
        self.llname = llname.format(c=self.city)
        self.is_rate = False
        if how == 'csv':
            if fname is None:
                self.fname = '{c}_box_radar_{yyyy}_{mm:02d}.csv'.format(c=self.city, yyyy=self.t.year, mm=self.t.month)
            else:
                self.fname = fname.format(c=self.city, yyyy=self.t.year, mm=self.t.month, dd=self.t.day)
            self.get_box(self.from_csv())

        if how == 'hdf5':
            self.store = store.format(c=self.city)
            if fname is None:
                if 'TOP50' in self.store:
                    fname = 'storm_{yyyy}_{mm:02d}_{dd:02d}'
                else:
                    fname = '{c}_{yyyy}_{mm:02d}'
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
        path = self.csv_path
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
        ll = pd.read_csv(self.llname, index_col=0, header=None, names=['lat','lon'])

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
                    if not np.isnan(np.sum(here)): 
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
    
    def add_buffer(self, p):
        from geopy.distance import vincenty

        edges = zip(self.lat[0, :], self.lon[0, :])
        edges.extend(zip(self.lat[:, -1], self.lon[:, -1]))
        edges.extend(zip(np.flipud(self.lat[-1, :]), np.flipud(self.lon[-1, :])))
        edges.extend(zip(np.flipud(self.lat[:, 0]), np.flipud(self.lon[:, 0])))
        
        for it in range(p.shape[0]):
            for ifeat in range(p.shape[1]):
                if np.isnan(p[it, ifeat, 'centroidY']):
                    continue
                center = p[it, ifeat, ['centroidY', 'centroidX']].values
                dist = min([vincenty(center, edge).kilometers for edge in edges])
                r = (p[it, ifeat, ['area']].values/np.pi)**.5
                if r>dist:
                    df0 = p[it,:,:]
                    for ichar in range(21):
                        df0.set_value(p.major_axis[ifeat], p.minor_axis[ichar], np.nan)
        return(p)

    def add_extra_buffer(self, p, extra):
        from geopy.distance import vincenty

        edges = zip(self.lat[0, :], self.lon[0, :])
        edges.extend(zip(self.lat[:, -1], self.lon[:, -1]))
        edges.extend(zip(np.flipud(self.lat[-1, :]), np.flipud(self.lon[-1, :])))
        edges.extend(zip(np.flipud(self.lat[:, 0]), np.flipud(self.lon[:, 0])))
        
        for it in range(p.shape[0]):
            for ifeat in range(p.shape[1]):
                if np.isnan(p[it, ifeat, 'centroidY']):
                    continue
                center = p[it, ifeat, ['centroidY', 'centroidX']].values
                dist = min([vincenty(center, edge).kilometers for edge in edges])
                r = (p[it, ifeat, ['area']].values/np.pi)**.5
                if r+extra>dist:
                    df0 = p[it,:,:]
                    for ichar in range(21):
                        df0.set_value(p.major_axis[ifeat], p.minor_axis[ichar], np.nan)
        return(p)

    def get_features(self, d={}, thresh=10, min_size=20, sigma=3, const=20, return_dict=True, buffer=False):
        '''
        Use r package SpatialVx to identify features. 
        
        Parameters
        ----------
       
        '''
        from rpy2 import robjects 
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        from pointprocess.common import import_r_tools, dotvars
        pandas2ri.activate()
        SpatialVx = importr('SpatialVx')
        rsummary = robjects.r.summary
        r_tools = import_r_tools()

        ll = np.array([self.lon.flatten('F'), self.lat.flatten('F')]).T
        for i in range(self.box.shape[0]-1):
            hold = SpatialVx.make_SpatialVx(self.box[i,:,:], self.box[i+1,:,:], loc=ll)
            look = r_tools.FeatureFinder_gaussian(hold, nx=self.box.shape[2], ny=self.box.shape[1], 
                                                  thresh=thresh, smoothpar=sigma, **(dotvars(min_size=min_size)))
            try:
                x = rsummary(look, silent=True)[0]
            except:
                continue
            px = pandas2ri.ri2py(x)
            df0 = pd.DataFrame(px, columns=['centroidX', 'centroidY', 'area', 'OrientationAngle', 
                                          'AspectRatio', 'Intensity0.25', 'Intensity0.9'])
            df0['Observed'] = list(df0.index+1)
            m = SpatialVx.centmatch(look, criteria=3, const=const)
            p = pandas2ri.ri2py(m[12])
            df1 = pd.DataFrame(p, columns=['Forecast', 'Observed'])
            l = SpatialVx.FeatureMatchAnalyzer(m)
            try:
                p = pandas2ri.ri2py(rsummary(l, silent=True))
            except:
                continue
            df2 = pd.DataFrame(p, columns=['Partial Hausdorff Distance','Mean Error Distance','Mean Square Error Distance',
                                          'Pratts Figure of Merit','Minimum Separation Distance', 'Centroid Distance',
                                          'Angle Difference','Area Ratio','Intersection Area','Bearing', 'Baddeleys Delta Metric',
                                          'Hausdorff Distance'])
            df3 = df1.join(df2)

            d.update({self.time[i]: pd.merge(df0, df3, how='outer')})
        if return_dict:
            return(d)
        p = pd.Panel(d)
        if buffer:
            return(self.add_buffer(p))
        return(p)

    def kde(self, lon, lat):
        import scipy.stats as st
        xx, yy = self.lon, self.lat
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([lon, lat])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        return(xx,yy,f)    
