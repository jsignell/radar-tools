import numpy as np

def get_z_from_radar(radar):
    """Input radar object, return z from radar (m, 2D)"""
    zz = []
    for sweep in range(radar.nsweeps):
        zz.append(radar.get_gate_x_y_z(sweep)[2])
    return np.concatenate(zz, axis=0)

def check_sounding_for_montonic(sounding):
    """
    So the sounding interpolation doesn't fail, force the sounding to behave
    monotonically so that z always increases. This eliminates data from
    descending balloons.
    """
    snd_T = sounding.soundingdata['temp']  # In old SkewT, was sounding.data
    snd_z = sounding.soundingdata['hght']  # In old SkewT, was sounding.data
    dummy_z = []
    dummy_T = []
    if not snd_T.mask[0]: #May cause issue for specific soundings
        dummy_z.append(snd_z[0])
        dummy_T.append(snd_T[0])
        for i, height in enumerate(snd_z):
            if i > 0:
                if snd_z[i] > snd_z[i-1] and not snd_T.mask[i]:
                    dummy_z.append(snd_z[i])
                    dummy_T.append(snd_T[i])
        snd_z = np.array(dummy_z)
        snd_T = np.array(dummy_T)
    return snd_T, snd_z

def interpolate_sounding_to_radar(sounding, radar):
    """Takes sounding data and interpolates it to every radar gate."""
    radar_z = get_z_from_radar(radar)
    radar_T = None
    snd_T, snd_z = check_sounding_for_montonic(sounding)
    shape = np.shape(radar_z)
    rad_z1d = radar_z.ravel()
    rad_T1d = np.interp(rad_z1d, snd_z, snd_T)
    return np.reshape(rad_T1d, shape), radar_z

def mask_data(radar, max_elevation_angle=0.6, min_dist_km=5, max_dist_km=300):
    # We just want the lowest sweeps and the gates at which we can reasonably have good data
    low_sweeps = [i for i, bool in enumerate(radar.fixed_angle['data']<max_elevation_angle) if bool]
    radar = radar.extract_sweeps(low_sweeps)

    if radar.range['units'] == 'meters':
        start_gate = (radar.range['data']< min_dist_km*1000).sum()
        end_gate = (radar.range['data']< max_dist_km*1000).sum()

    for field in radar.fields.keys():
        radar.fields[field]['data'].mask[:,:start_gate] = True
    return radar, start_gate, end_gate

def separate_dp_vel_sweeps(radar):
    # check that adjoining times are close?
    dp_sweeps = [i for i in range(radar.nsweeps) if ~radar.extract_sweeps([i]).fields['differential_phase']['data'].mask.all()]
    vel_sweeps = [i for i in range(radar.nsweeps) if ~radar.extract_sweeps([i]).fields['velocity']['data'].mask.all()]

    vel_radar = radar.extract_sweeps(vel_sweeps)
    dp_radar = radar.extract_sweeps(dp_sweeps)
    
    return dp_radar, vel_radar

def construct_sw_vel_mask(vel_radar):
    vel = vel_radar.fields['velocity']['data']
    sw = vel_radar.fields['spectrum_width']['data']

    # construct a mask with the same shape as vel, but all False
    sw_vel_mask = np.zeros_like(vel.mask)

    # mask places where neither vel nor sw are masked, and both are zero
    rows_cols  = np.stack(np.where((~vel.mask) & (~sw.mask) & (vel==0) & (sw==0)), axis=1)
    for row, col in rows_cols:
        sw_vel_mask[row, col:] = True
    return sw_vel_mask

def mask_dp_fiels(vel_radar, dp_radar, sw_vel_mask):
    # add time check? 
    v = np.append(vel_radar.sweep_start_ray_index['data'], None)
    d = np.append(dp_radar.sweep_start_ray_index['data'], None)
    for i in range(vel_radar.nsweeps):
        a = (np.abs(dp_radar.azimuth['data'][d[i]:d[i+1]]-vel_radar.azimuth['data'][v[i]])).argmin()
        mask = np.roll(sw_vel_mask[v[i]:v[i+1]], a)
        for field in ['reflectivity', 'differential_reflectivity',
                      'cross_correlation_ratio', 'differential_phase']:
            dp_radar.fields[field]['data'].mask[d[i]:d[i+1]]+= mask
    return dp_radar

def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID',
                              dz_field='reflectivity'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

def extract_unmasked_data(radar, field, bad=-32768):
    """Simplify getting unmasked radar fields from Py-ART"""
    return radar.fields[field]['data'].filled(fill_value=bad)

def interpolate_radially(radar, field, start_gate, end_gate, interpolate_max=10):
    '''
    Interpolate first around rings, then interpolate along axials
    
    Parameters
    ----------
    radar: pyart radar object
    field: str representing radar field to interpolate
    start_gate: index of first gate that we will use
    end_gate: index of last gate that we will use
    interpolate_max: max distance in grid cells to allow interpolation
    
    Returns
    -------
    radar: with field interpolated and re-masked
    '''
    rain = radar.fields[field]['data']
    sweep_starts = np.append(radar.sweep_start_ray_index['data'],
                             radar.sweep_end_ray_index['data'][-1]+1)

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def func(a, interpolate_max):
        # for each ring in the sweep we will try to interpolate
        x=np.arange(a.shape[0])
        y=a.data
        ma0=a.mask
        ma1 = np.sum(rolling_window(ma0, interpolate_max), axis=1)<interpolate_max
        ma1 = np.concatenate([ma1, np.array([ma1[-1]]*(interpolate_max-1))])
        y[ma0 & ma1] = np.interp(x[ma0 & ma1], x[~ma0], y[~ma0], left=0, right=0)
        a.mask = ma0 & ~ma1

    for n in range(len(sweep_starts)-1):
        foo = [func(rain[sweep_starts[n]:sweep_starts[n+1],i], interpolate_max) for
               i in range(start_gate, end_gate)]
        foo = [func(rain[i, start_gate:end_gate], interpolate_max) for
               i in range(sweep_starts[n],sweep_starts[n+1])]
    return radar

def get_kdp(radar, thsd=12, window=3, 
            radar_field_dict=dict(field_name='kdp', units='degrees/km',
                                  long_name='Specific differential phase', 
                                  standard_name='specific_differential_phase')):
    from csu_radartools import csu_kdp

    dz = extract_unmasked_data(radar, 'reflectivity')
    dp = extract_unmasked_data(radar, 'differential_phase')
    # Range needs to be supplied as a variable, and it needs to be the same shape as dzN, etc.
    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])

    gs = rng2d[0,1] - rng2d[0,0]
    kd, fd, sd = csu_kdp.calc_kdp_bringi(dp=dp, dz=dz, rng=rng2d/1000.0, thsd=thsd, gs=gs, window=window)
    radar = add_field_to_radar_object(kd, radar, **radar_field_dict)
    return radar

def calculate_hidro_rain(radar, sounding, 
                           rain_field_dict=dict(field_name='rain', units='mm h-1',
                                                long_name='HIDRO Rainfall Rate', 
                                                standard_name='rain_hca'),
                           method_field_dict=dict(field_name='method', units='',
                                                  long_name='HIDRO Rainfall Method', 
                                                  standard_name='rain_estimate_method')):
    '''
    Calculate rainfall using the hidro_rain, method
    
    Parameters
    ----------
    radar: pyart radar object from nexrad volume scan
    sounding: SkewT sounding for the appropriate time and location
    
    Returns
    -------
    radar: with new rain field added according to dict
    '''
    from csu_radartools import csu_fhc, csu_blended_rain
    dz = extract_unmasked_data(radar, 'reflectivity')
    dr = extract_unmasked_data(radar, 'differential_reflectivity')
    dp = extract_unmasked_data(radar, 'differential_phase')
    rh = extract_unmasked_data(radar, 'cross_correlation_ratio')

    radar_T, radar_z = interpolate_sounding_to_radar(sounding, radar)
    if 'kdp' not in radar.fields.keys():
        radar = get_kdp(radar)
    kd = extract_unmasked_data(radar, 'kdp')
    scores = csu_fhc.csu_fhc_summer(dz=dz, zdr=dr, rho=rh, kdp=kd, band='S', use_temp=True, T=radar_T)
    fh = np.argmax(scores, axis=0) + 1

    rain, method = csu_blended_rain.csu_hidro_rain(dz=dz, zdr=dr, kdp=kd, fhc=fh)
    radar = add_field_to_radar_object(rain, radar, **rain_field_dict)
    radar = add_field_to_radar_object(method, radar, **method_field_dict)
    return radar

def calculate_rain_kdp(radar, b=0.87,
                       rain_field_dict=dict(field_name='r_kdp', units='mm h-1',  
                                            long_name='Rainfall Rate R(Kdp)', 
                                            standard_name='rain_kdp')):
    '''
    Calculate rainfall using the rain_kdp method
    
    Parameters
    ----------
    radar: pyart radar object from nexrad volume scan
    
    Returns
    -------
    radar: with new rain field added according to dict
    '''
    from csu_radartools import csu_blended_rain
    if 'kdp' not in radar.fields.keys():
        radar = get_kdp(radar)
    kd = extract_unmasked_data(radar, 'kdp')
    r_kdp = csu_blended_rain.calc_rain_kdp(kd, b=b)
    radar = add_field_to_radar_object(r_kdp, radar, **rain_field_dict)
    return radar

def calculate_rain_nexrad(radar,
                          rain_field_dict=dict(field_name='r_z', units='mm h-1',
                                               long_name='Rainfall Rate R(Z)', 
                                               standard_name='rain_z')):
    '''
    Calculate rainfall using the rain_nexrad_r method
    
    Parameters
    ----------
    radar: pyart radar object from nexrad volume scan
    
    Returns
    -------
    radar: with new rain field added according to dict
    '''
    from csu_radartools import csu_blended_rain

    dz = extract_unmasked_data(radar, 'reflectivity')
    r_z = csu_blended_rain.calc_rain_nexrad(dz)
    radar = add_field_to_radar_object(r_z, radar, **rain_field_dict)
    return radar

def find_x_y_displacement(radar, longitude, latitude):
    """ Return the x and y displacement (in meters) from a radar location. """
    import pyproj

    # longitude and latitude in degrees
    lat_0 = radar.latitude['data'][0]
    lon_0 = radar.longitude['data'][0]
    proj = pyproj.Proj(proj='aeqd', lon_0=lon_0, lat_0=lat_0)
    return proj(longitude, latitude)