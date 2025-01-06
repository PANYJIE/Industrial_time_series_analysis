# -*- coding: utf-8 -*-



from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
# tf.reset_default_graph()
import random
import feather
import argparse
import os
from keras import backend as K




# random.seed(9001)
# tf.set_random_seed(9001)
# np.random.seed(9001)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '/device:CPU:0'

SEED = 9001
os.environ['PYTHONHASHSEED']=str(SEED)
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

''' Define Physics Helper Functions '''

def transformTempToDensity(temp):
    densities = 1000 * (1 - ((temp + 288.9414) * tf.pow(temp - 3.9863, 2)) / (508929.2 * (temp + 68.12963)))
    return densities

def den_loss(temp):
    den = transformTempToDensity(temp)
    den_d1 = den[0: - 1, :]
    den_d2 = den[1:, :]
    d_loss = tf.reduce_mean(tf.nn.relu(den_d1 - den_d2))
    return d_loss

def calculate_ec_loss(inputs, outputs, phys, depth_areas, n_depths, ec_threshold, combine_days=1):
    densities = transformTempToDensity(outputs)

    diff_per_set = []
    # loop through sets of n_depths
    for i in range(1):  #
        # indices
        start_index = (i) * n_depths
        end_index = (i + 1) * n_depths

        # calculate lake energy for each timestep，
        lake_energies = calculate_lake_energy(
            outputs[start_index:end_index, :], densities[start_index:end_index, :], depth_areas,n_depths)

        # calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]

        # calculate sum of energy flux into or out of the lake at each timestep
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index, :, :], outputs[start_index, :],
                                                     combine_days)  #
        diff_vec = tf.abs(lake_energy_deltas - lake_energy_fluxes)

        tmp_mask = 1 - phys[start_index + 1, 1:-1, 9]
        tmp_loss = tf.reduce_mean(diff_vec * tf.cast(tmp_mask, tf.float32))
        diff_per_set.append(tmp_loss)

    diff_per_set_r = tf.stack(diff_per_set)

    diff_per_set = tf.clip_by_value(diff_per_set_r - ec_threshold, clip_value_min=0, clip_value_max=999999)

    return tf.reduce_mean(diff_per_set), diff_vec, diff_per_set_r, diff_per_set

def calculate_lake_energy(temps, densities, depth_areas,n_depths):
    # calculate the total energy of the lake for every timestep
    # sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    # then multiply by the specific heat of water
    dz = 0.5  # thickness for each layer
    cw = 4186  # specific heat of water
    depth_areas = tf.reshape(depth_areas, [n_depths, 1])
    energy = tf.reduce_sum(tf.multiply(tf.cast(depth_areas, tf.float32), temps) * densities * dz * cw, 0)
    return energy

def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    # given a time series of energies, compute and return the differences
    # between each time step
    time = 86400  # seconds per day
    energy_deltas = (energies[1:] - energies[:-1]) / time / surface_area
    return energy_deltas

def calculate_air_density(air_temp, rh):
    # returns air density in kg / m^3
    # equation from page 13 GLM/GLEON paper(et al Hipsey)

    # Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    # atmospheric pressure
    p = 1013.  # mb

    # water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)

    # water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure / (p - vapPressure)
    return (1.0 / c_gas * (1 + r) / (1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100

def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):  #
    # equation 22 in GLM/GLEON paper(et al Hipsey)
    # GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    # calculate air density
    rho_a = calculate_air_density(air_temp, rel_hum)

    # specific heat capacity of air in J/(kg*C)
    c_a = 1005.

    # bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    # wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)

    return -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):  #
    # equation 23 in GLM/GLEON paper(et al Hipsey)
    # GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    # air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    # bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    # latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    # wind speed at 10m height
    U_10 = calculate_wind_speed_10m(wind_speed)

    # ratio of molecular weight of water to that of dry air
    omega = 0.622

    # air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)

def calculate_vapour_pressure_air(rel_hum, temp):  #
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)

def calculate_vapour_pressure_saturated(temp):  #
    # returns in millibars
    exponent = (9.28603523 - (2332.37885 / (temp + 273.15))) * np.log(10)
    return tf.exp(exponent)

def calculate_wind_speed_10m(ws, ref_height=2.):  #
    # from GLM code glm_surface.c
    c_z0 = 0.001  # default roughness
    return ws * (tf.log(10.0 / c_z0) / tf.log(ref_height / c_z0))

def calculate_energy_fluxes(phys, surf_temps, combine_days):  #
    e_s = 0.985  # emissivity of water  es
    alpha_sw = 0.07  # shortwave albedo   asw
    alpha_lw = 0.03  # longwave albedo   alw
    sigma = 5.67e-8  # Stefan-Baltzmann constant   δ
    R_sw_arr = phys[:-1, 2] + (phys[1:, 2] - phys[:-1, 2]) / 2  # Rsw
    R_lw_arr = phys[:-1, 3] + (phys[1:, 3] - phys[:-1, 3]) / 2  # RLwin
    R_lw_out_arr = e_s * sigma * (tf.pow(surf_temps[:] + 273.15, 4))  # Rlwout
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:] - R_lw_out_arr[:-1]) / 2  # Rlwout

    air_temp = phys[:-1, 4]
    air_temp2 = phys[1:, 4]
    rel_hum = phys[:-1, 5]
    rel_hum2 = phys[1:, 5]
    ws = phys[:-1, 6]
    ws2 = phys[1:, 6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2) / 2  #
    H = (H + H2) / 2

    fluxes = (R_sw_arr[:-1] * (1 - alpha_sw) + R_lw_arr[:-1] * (1 - alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])
    return fluxes

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)






''' Load data '''
def train(ii,data_path,save_path,preds_path,lr,label_train,epoch):
    # learning_rate = 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=data_path)

    # parser.add_argument('--restore_path', default='improve/tmp/mendo/model1')
    parser.add_argument('--save_path', default=save_path)
    parser.add_argument('--preds_path', default=preds_path)
    args = parser.parse_args()

    tf.reset_default_graph()
    set_global_determinism(seed=SEED)

    learning_rate = lr
    epochs = epoch  #
    state_size = 20
    input_size = 10
    phy_size = 10  #
    n_steps = 353  #
    n_classes = 1
    N_sec = 19
    elam = 0.005
    # dlam = 1
    ec_threshold = 24  #
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=data_path)

    # parser.add_argument('--restore_path', default='improve/tmp/mendo/model1')
    # parser.add_argument('--save_path', default=save_path)
    # parser.add_argument('--preds_path', default=preds_path)
    args = parser.parse_args()

    ''' Declare constant hyperparameters '''

    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    ''' Define Graph '''

    x = tf.placeholder("float", [None, n_steps, input_size])
    x_d = tf.placeholder("float", [None, n_depths, input_size])
    y = tf.placeholder("float", [None, n_steps])
    # y_d = tf.placeholder("float", [None, n_steps])
    y_phy = tf.placeholder("float", [None, n_steps])
    m = tf.placeholder("float", [None, n_steps])
    bt_sz = tf.placeholder("int32", None)
    x_u = tf.placeholder("float", [None, n_steps, input_size])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)
    lstm_cell_depth = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)

    # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=drop_radio)
    # lstm_cell_depth = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_depth, output_keep_prob=drop_radio)

    # （outputs, states），outputs=【batchsize,timestamp,cell_num】,states=[2,batchsize,cell_num]
    with tf.variable_scope("rnn1") as scope_sp:
        state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    with tf.variable_scope("rnn2") as scope_sp1:
        state_series_x_depth, current_state_x_depth = tf.nn.dynamic_rnn(lstm_cell_depth, x_d, dtype=tf.float32)

    w_fin = tf.get_variable('w_fin', [state_size, n_classes], tf.float32, tf.random_normal_initializer(stddev=0.02))
    b_fin = tf.get_variable('b_fin', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    # print(w_fin)

    w_fin_d = tf.get_variable('w_fin_d', [state_size, n_classes], tf.float32, tf.random_normal_initializer(stddev=0.02))
    b_fin_d = tf.get_variable('b_fin_d', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    # print(w_fin_d)

    w1 = tf.get_variable('w1', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable('w2', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    w3 = tf.get_variable('w3', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    b1 = tf.get_variable('b1', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    keep_prob = 0.5

    pred = []
    for i in range(n_steps):
        tp1 = state_series_x[:, i, :]
        # print(tp1)
        pt = tf.matmul(tp1, w_fin) + b_fin
        # pt = tf.nn.dropout(pt, keep_prob)
        # print(pt)
        pred.append(pt)
    # print(np.array(pred).shape)
    # print(pred)

    #
    pred = tf.stack(pred, axis=1)
    # print(pred.shape)
    pred_s = tf.reshape(pred, [-1, 1])
    # print(np.array(pred_s).shape)
    # print(pred_s)

    pred_d = []
    for i in range(n_depths):
        tp1_d = state_series_x_depth[:, i, :]
        # print(tp1_d)
        pt_d = tf.matmul(tp1_d, w_fin_d) + b_fin_d
        # pt_d = tf.nn.dropout(pt_d, keep_prob)
        # print(pt_d)
        pred_d.append(pt_d)
    # print(np.array(pred_d).shape)
    # print(pred_d)
    pred_s_d = tf.reshape(pred_d, [-1, 1])
    # print(np.array(pred_s_d).shape)
    # print(pred_s_d)

    y_s_phy = tf.reshape(y_phy, [-1, 1])
    # print(np.array(y_s_phy).shape)
    # print(y_s_phy)

    # pred_s_all = (pred_s+pred_s_d+y_s_phy)/3
    # pred_s_all = pred_s+pred_s_d+y_s_phy
    # pred_s_all = pred_s/3
    pred_s_all2 = w1 * pred_s + w2 * pred_s_d + w3 * y_s_phy + b1
    pred_s_all2_old = tf.reshape(pred_s_all2, [50, 353, 1])

    y_s = tf.reshape(y, [-1, 1])
    m_s = tf.reshape(m, [-1, 1])

    # r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s))
    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s_all2 - y_s), m_s))) / tf.reduce_sum(m_s))
    # r_cost = K.sqrt(K.sum(K.square((pred_s-y_s)*m_s))/K.sum(m_s))
    # r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s_all2 - y_s), m_s))) / tf.reduce_sum(m_s))

    ''' Continue Graph Definition '''

    unsup_inputs = tf.placeholder("float", [None, n_steps, input_size])  #
    un_x_d = tf.placeholder("float", [None, n_depths, input_size])  #
    un_y_phy = tf.placeholder("float", [None, n_steps])  #

    with tf.variable_scope("rnn1", reuse=True) as scope_sp:
        state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, unsup_inputs, dtype=tf.float32, scope=scope_sp)

    with tf.variable_scope("rnn2", reuse=True) as scope_sp1:
        state_series_xu_d, current_state_xu_d = tf.nn.dynamic_rnn(lstm_cell_depth, un_x_d, dtype=tf.float32,
                                                                  scope=scope_sp)

    keep_prob = 0.5
    pred_u = []
    for i in range(n_steps):
        tp2 = state_series_xu[:, i, :]
        pt2 = tf.matmul(tp2, w_fin) + b_fin
        # pt2 = tf.nn.dropout(pt2,keep_prob)
        pred_u.append(pt2)

    pred_u = tf.stack(pred_u, axis=1)  #
    pred_u = tf.reshape(pred_u, [-1, n_steps])  #

    pred_d_u = []
    for i in range(n_depths):
        tp1_d_u = state_series_xu_d[:, i, :]  #
        # print(tp1_d_u)
        pt_d_u = tf.matmul(tp1_d_u, w_fin_d) + b_fin_d  #
        # pt_d_u = tf.nn.dropout(pt_d_u, keep_prob)
        # print(pt_d_u)
        pred_d_u.append(pt_d_u)  #
    # print(np.array(pred_d_u).shape)  #
    # print(pred_d_u)
    pred_s_d_u = tf.reshape(pred_d_u, [-1, n_steps])

    y_s_phy_u = tf.reshape(un_y_phy, [-1, n_steps])  #

    # unpred_s_all = (pred_u+pred_s_d_u+y_s_phy_u)/3
    # unpred_s_all = pred_u+pred_s_d_u+y_s_phy_u
    unpred_s_all2 = w1 * pred_u + w2 * pred_s_d_u + w3 * y_s_phy_u + b1

    unsup_phys_data = tf.placeholder("float", [None, n_steps, phy_size])
    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    unsup_loss, a, b, c = calculate_ec_loss(unsup_inputs,
                                            unpred_s_all2,
                                            unsup_phys_data,
                                            depth_areas,
                                            n_depths,
                                            ec_threshold,
                                            combine_days=1)

    # d_loss = den_loss(unpred_s_all)
    cost = r_cost + elam * unsup_loss
    # cost = r_cost + elam * unsup_loss +dlam*d_loss

    tvars = tf.trainable_variables()

    # regularization_cost = l2_radio * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tvars ])
    # cost = r_cost + elam * unsup_loss+ regularization_cost
    # for i in tvars:
    #     print(i)
    grads = tf.gradients(cost, tvars)

    saver = tf.train.Saver(max_to_keep=5)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))




    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    print("run "+str(ii)+" times")



    x_full = np.load(os.path.join(args.data_path, 'processed_features'+str(ii)+'.npy'))
    yphy = np.load(os.path.join(args.data_path, 'labels_phy'+str(ii)+'.npy'))
    x_raw_full = np.load(os.path.join(args.data_path, 'features'+str(ii)+'.npy'))
    diag_full = np.load(os.path.join(args.data_path, 'diag.npy'))

    # ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Daily.Qe', 'Daily.Qh', 'Has.Black.Ice']
    phy_full = np.concatenate((x_raw_full[:,:,:-3],diag_full),axis=2)

    new_dates = np.load(os.path.join(args.data_path, 'dates.npy'), allow_pickle=True)

    train_data = feather.read_dataframe(os.path.join(args.data_path, label_train))

    tr_date = train_data.values[:,0]
    tr_depth = train_data.values[:,1]
    tr_temp = train_data.values[:,2]


    t_steps = x_raw_full.shape[1]#3549
    # print(t_steps)
    m_tr = np.zeros([n_depths,t_steps])#50*3549
    obs_tr = np.zeros([n_depths,t_steps])
    k=0
    #dd = 0
    for i in range(new_dates.shape[0]):#3550
        if k>=tr_date.shape[0]:
            break
        # print(new_dates[i])
        # print(tr_date[k])
        while new_dates[i]==tr_date[k]:
            d = min(int(tr_depth[k]/0.5),n_depths-1)
            m_tr[d,i]=1
            obs_tr[d,i]=tr_temp[k]
            k+=1
            if k>=tr_date.shape[0]:
                break

    test_data = feather.read_dataframe(os.path.join(args.data_path, 'labels_test'+str(ii)+'.feather'))

    te_date = test_data.values[:,0]
    te_depth = test_data.values[:,1]
    te_temp = test_data.values[:,2]

    m_te = np.zeros([n_depths,t_steps])
    obs_te = np.zeros([n_depths,t_steps])
    k=0
    #dd = 0
    for i in range(new_dates.shape[0]):
        # print(k)
        # print(te_date.shape)
        if k>=te_date.shape[0]:
            break
        # print(new_dates[i])
        # print(te_date[k])
        while new_dates[i]==te_date[k]:
            d = min(int(te_depth[k]/0.5),n_depths-1)
    #        if m_te[d,i]==1:
    #            print(d,te_depth[k])
            m_te[d,i]=1
            obs_te[d,i]=te_temp[k]
            # print(te_temp[k])
            k+=1
            if k>=te_date.shape[0]:
                break
    # for i in range(0,obs_te.shape[1]):
    #     print(obs_te[0,i])

    x_train = np.zeros([n_depths*N_sec,n_steps,input_size])
    x_train_d = np.zeros([n_steps*N_sec,n_depths,input_size])
    y_train_phy = np.zeros([n_depths*N_sec,n_steps])
    y_train = np.zeros([n_depths*N_sec,n_steps])
    p_train = np.zeros([n_depths*N_sec,n_steps,phy_size])
    m_train = np.zeros([n_depths*N_sec,n_steps])
    y_test = np.zeros([n_depths*N_sec,n_steps])
    m_test = np.zeros([n_depths*N_sec,n_steps])

    x_full_d = x_full.transpose(1,0,2)

    for i in range(1,N_sec+1):
        x_train[(i-1)*n_depths:i*n_depths,:,:]=x_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
        y_train_phy[(i - 1) * n_depths:i * n_depths, :] = yphy[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
        y_train[(i-1)*n_depths:i*n_depths,:]=obs_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        p_train[(i-1)*n_depths:i*n_depths,:,:]=phy_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
        m_train[(i-1)*n_depths:i*n_depths,:]=m_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]

        y_test[(i-1)*n_depths:i*n_depths,:]=obs_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        m_test[(i-1)*n_depths:i*n_depths,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        x_train_d[(i-1)*n_steps:i*n_steps,:,:] = x_full_d[int((i-1)*n_steps/2):int((i+1)*n_steps/2),:,:]



    # print(np.array(x_train).shape)
    # print(np.array(y_train).shape)
    # print(x_train[0,:,:])
    # print(x_train[50,:,:])
    # print(x_train[48,:,:])
    # print(y_train[0,:])
    # print(y_train[48,:])
    # print(y_train[50,:])
    # print(m_train[0,:])
    # print(m_train[48,:])
    # print(m_train[50,:])
    # print(y_test[0,:])
    # print(y_test[48,:])
    # print(y_test[450,:])
    ''' Train '''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # If using pretrained model, reload it now
        # if args.restore_path != '':
        #     saver.restore(sess, os.path.join(args.restore_path, 'trained_model_pre+_1_400.ckpt'))
        rc_list = []
        for epoch in range(epochs):
            loss = 0
            rc = 0
            ec = 0
            count = 0
            for i in range(1, N_sec+1):#12的时候mask为0

                x_train_b = x_train[(i-1)*n_depths:i*n_depths,:,:]
                x_train_d_b = x_train_d[(i-1)*n_steps:i*n_steps,:,:]
                y_train_phy_b = y_train_phy[(i - 1) * n_depths:i * n_depths, :]
                y_train_b = y_train[(i-1)*n_depths:i*n_depths,:]
                m_train_b = m_train[(i-1)*n_depths:i*n_depths,:]
                x_f = x_train_b
                x_f_d = x_train_d_b
                y_f_phy = y_train_phy_b
                p_f = p_train[(i-1)*n_depths:i*n_depths,:,:]
                # print(np.array(x_train_b).shape)
                # print(np.array(x_train_d_b).shape)
                # print(np.array(y_train_phy_b).shape)
                # print(np.array(y_train_b).shape)
                # print(np.array(m_train_b).shape)
                # print(np.array(p_f).shape)

                tt = np.sum(m_train_b, axis=0)
                aaaa=np.sum(tt,axis=0)
                if aaaa!=0:

                    _, loss_,rc_,ec_,aa,bb,cc,prd1 = sess.run(
                            [train_op, cost,r_cost,unsup_loss,a,b,c,pred_s_all2_old],
                            feed_dict = {
                                    x: x_train_b,
                                    x_d:x_train_d_b,
                                    y_phy: y_train_phy_b,
                                    y: y_train_b,
                                    m: m_train_b,
                                    unsup_inputs: x_f,
                                    un_x_d:x_f_d,
                                    un_y_phy:y_f_phy,
                                    unsup_phys_data: p_f,
                                    bt_sz: n_depths*N_sec
                        })
                    print("batch " + str(i) + ", BatLoss= " + \
                          "{:.32f}".format(loss_) + ", Rc= " + \
                          "{:.32f}".format(rc_) + ", Ec= " + \
                          "{:.32f}".format(ec_))
                    loss = loss +loss_
                    rc = rc + rc_
                    ec = ec + ec_
                    count = count+1
                # with tf.Session() as sess:
                #     print(sess.run(aaaa))
            loss = loss/count
            rc = rc/count
            ec = ec/count
            rc_list.append(rc)
            if epoch%1==0:
                print("Step " + str(epoch) + ", BatLoss= " + \
                  "{:.32f}".format(loss) + ", Rc= " + \
                  "{:.32f}".format(rc) + ", Ec= " + \
                  "{:.32f}".format(ec))
            if (epoch+1)%100 == 0:
                loss_te = 0
                Ec_te = 0
                count_te = 0
                for i in range(1, N_sec + 1):
                    x_train_b = x_train[(i - 1) * n_depths:i * n_depths, :, :]
                    x_train_d_b = x_train_d[(i - 1) * n_steps:i * n_steps, :, :]
                    y_train_phy_b = y_train_phy[(i - 1) * n_depths:i * n_depths, :]
                    y_test_b = y_test[(i - 1) * n_depths:i * n_depths, :]
                    m_test_b = m_test[(i - 1) * n_depths:i * n_depths, :]
                    x_f = x_train_b
                    x_f_d = x_train_d_b
                    y_f_phy = y_train_phy_b
                    p_f = p_train[(i - 1) * n_depths:i * n_depths, :, :]
                    tt = np.sum(m_test_b, axis=0)
                    aaaa2 = np.sum(tt, axis=0)
                    # aaaa2 = tf.reduce_sum(tf.reshape(m_test_b, [-1, 1]))
                    loss_te_, Ec_te_, prd = sess.run([r_cost, unsup_loss, pred_s_all2_old],
                                                     feed_dict={x: x_train_b, x_d: x_train_d_b, y_phy: y_train_phy_b,
                                                                y: y_test_b,
                                                                m: m_test_b, unsup_inputs: x_f, un_x_d: x_f_d,
                                                                un_y_phy: y_f_phy,
                                                                unsup_phys_data: p_f, bt_sz: n_depths * N_sec})
                    if i == 1:
                        prd_t = prd
                    if i > 1:
                        prd_t = np.vstack((prd_t, prd))
                    if aaaa2 != 0:
                        # loss_te_,Ec_te_,prd = sess.run([r_cost,unsup_loss,pred], feed_dict = {x: x_train_b,x_d:x_train_d_b,y_phy: y_train_phy_b, y: y_test_b, m: m_test_b,unsup_inputs: x_f,un_x_d:x_f_d,un_y_phy:y_f_phy,unsup_phys_data: p_f, bt_sz: n_depths*N_sec})

                        loss_te = loss_te + loss_te_
                        Ec_te = Ec_te + Ec_te_
                        count_te = count_te + 1
                        print("batch " + str(i))
                        print("Loss_te " + "{:.4f}".format(loss_te_))
                        print("Ec= " + "{:.4f}".format(Ec_te_))
                loss_te = loss_te / count_te
                Ec_te = Ec_te / count_te
                print("Loss_te_mean " + "{:.4f}".format(loss_te))
                print("Ec_mean= " + "{:.4f}".format(Ec_te))
                # print(dd)
                # if args.save_path != '':
                # current_dir_pid = os.path.dirname(__file__)
                # target_dir1_pid = os.path.join(current_dir_pid)
                # saver.save(sess, os.path.join(target_dir1_pid , save_path ,  "trained_model2_+_" + str(ii) + "_" + str(epoch + 1) + ".ckpt"))
                saver.save(sess, os.path.join(save_path, "trained_model2_+_" + str(ii) + "_" + str(epoch + 1) + ".ckpt"))
                # saver.save(sess, os.path.join(save_path, "trained_model2_+_"+str(ii)+"_"+str(epoch+1)+".ckpt"))

                # predict on test data, reshape to output file format, and save
                # loss_te,prd = sess.run([r_cost,pred], feed_dict = {x: x_train,x_d:x_train_d,y_phy: y_train_phy,y: y_test, m: m_test})
                # print("Loss_te " + "{:.4f}".format(loss_te))
                prd = prd_t
                prd_o = np.zeros([n_depths, n_steps + int((N_sec - 1) * n_steps / 2)])
                prd_o[:, :n_steps] = prd[0:n_depths, :, 0]
                for j in range(N_sec - 1):
                    st_idx = n_steps - (int((j + 1) * n_steps / 2) - int(j * n_steps / 2))  # handle even or odd cases
                    prd_o[:, n_steps + int(j * n_steps / 2):n_steps + int((j + 1) * n_steps / 2)] = prd[(j + 1) * n_depths:(
                                                                                                                                       j + 2) * n_depths,
                                                                                                    st_idx:, 0]
                prd_o_ = np.transpose(prd_o)
                np.savetxt(os.path.join(preds_path, "predict_model2_+_"+str(ii)+"_"+str(epoch+1)+".csv"), prd_o, delimiter=',')
                # np.savetxt(os.path.join(args.preds_path, "predict_pgdl55.csv"), prd_o_, delimiter=',')
        # current_dir_pid_preds = os.path.dirname(__file__)
        # target_dir1_pid_preds = os.path.join(current_dir_pid_preds)
        # rc_np = np.array(rc_list)
        # np.savetxt(os.path.join(target_dir1_pid_preds, preds_path, "train_loss2" + str(ii) + ".csv"), rc_np, delimiter=',')
        # rc_np =  np.array(rc_list)
        # np.savetxt(os.path.join(preds_path, "train_loss2" + str(ii) + ".csv"), rc_np, delimiter=',')

def test(ii,data_path,save_path,preds_path,lr,label_test,epoch):
    # learning_rate = 0.001
    test = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=data_path)

    # parser.add_argument('--restore_path', default='improve/tmp/mendo/model1')
    parser.add_argument('--save_path', default=save_path)
    parser.add_argument('--preds_path', default=preds_path)
    args = parser.parse_args()

    tf.reset_default_graph()
    set_global_determinism(seed=SEED)

    learning_rate = lr
    epochs = epoch  #
    state_size = 20
    input_size = 10
    phy_size = 10  #
    n_steps = 353  #
    n_classes = 1
    N_sec = 19
    elam = 0.005
    # dlam = 1
    ec_threshold = 24  #
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=data_path)

    # parser.add_argument('--restore_path', default='improve/tmp/mendo/model1')
    # parser.add_argument('--save_path', default=save_path)
    # parser.add_argument('--preds_path', default=preds_path)
    args = parser.parse_args()

    ''' Declare constant hyperparameters '''

    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    ''' Define Graph '''

    x = tf.placeholder("float", [None, n_steps, input_size])
    x_d = tf.placeholder("float", [None, n_depths, input_size])
    y = tf.placeholder("float", [None, n_steps])
    # y_d = tf.placeholder("float", [None, n_steps])
    y_phy = tf.placeholder("float", [None, n_steps])
    m = tf.placeholder("float", [None, n_steps])
    bt_sz = tf.placeholder("int32", None)
    x_u = tf.placeholder("float", [None, n_steps, input_size])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)
    lstm_cell_depth = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)

    # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=drop_radio)
    # lstm_cell_depth = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_depth, output_keep_prob=drop_radio)

    # （outputs, states），outputs=【batchsize,timestamp,cell_num】,states=[2,batchsize,cell_num]
    with tf.variable_scope("rnn1") as scope_sp:
        state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    with tf.variable_scope("rnn2") as scope_sp1:
        state_series_x_depth, current_state_x_depth = tf.nn.dynamic_rnn(lstm_cell_depth, x_d, dtype=tf.float32)

    w_fin = tf.get_variable('w_fin', [state_size, n_classes], tf.float32, tf.random_normal_initializer(stddev=0.02))
    b_fin = tf.get_variable('b_fin', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    # print(w_fin)

    w_fin_d = tf.get_variable('w_fin_d', [state_size, n_classes], tf.float32, tf.random_normal_initializer(stddev=0.02))
    b_fin_d = tf.get_variable('b_fin_d', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    # print(w_fin_d)

    w1 = tf.get_variable('w1', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable('w2', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    w3 = tf.get_variable('w3', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    b1 = tf.get_variable('b1', [n_classes], tf.float32, initializer=tf.constant_initializer(0.0))
    keep_prob = 0.5

    pred = []
    for i in range(n_steps):
        tp1 = state_series_x[:, i, :]
        # print(tp1)
        pt = tf.matmul(tp1, w_fin) + b_fin
        # pt = tf.nn.dropout(pt, keep_prob)
        # print(pt)
        pred.append(pt)
    # print(np.array(pred).shape)
    # print(pred)

    #
    pred = tf.stack(pred, axis=1)
    # print(pred.shape)
    pred_s = tf.reshape(pred, [-1, 1])
    # print(np.array(pred_s).shape)
    # print(pred_s)

    pred_d = []
    for i in range(n_depths):
        tp1_d = state_series_x_depth[:, i, :]
        # print(tp1_d)
        pt_d = tf.matmul(tp1_d, w_fin_d) + b_fin_d
        # pt_d = tf.nn.dropout(pt_d, keep_prob)
        # print(pt_d)
        pred_d.append(pt_d)
    # print(np.array(pred_d).shape)
    # print(pred_d)
    pred_s_d = tf.reshape(pred_d, [-1, 1])
    # print(np.array(pred_s_d).shape)
    # print(pred_s_d)

    y_s_phy = tf.reshape(y_phy, [-1, 1])
    # print(np.array(y_s_phy).shape)
    # print(y_s_phy)

    # pred_s_all = (pred_s+pred_s_d+y_s_phy)/3
    # pred_s_all = pred_s+pred_s_d+y_s_phy
    # pred_s_all = pred_s/3
    pred_s_all2 = w1 * pred_s + w2 * pred_s_d + w3 * y_s_phy + b1
    pred_s_all2_old = tf.reshape(pred_s_all2, [50, 353, 1])

    y_s = tf.reshape(y, [-1, 1])
    m_s = tf.reshape(m, [-1, 1])

    # r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s))
    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s_all2 - y_s), m_s))) / tf.reduce_sum(m_s))
    # r_cost = K.sqrt(K.sum(K.square((pred_s-y_s)*m_s))/K.sum(m_s))
    # r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s_all2 - y_s), m_s))) / tf.reduce_sum(m_s))

    ''' Continue Graph Definition '''

    unsup_inputs = tf.placeholder("float", [None, n_steps, input_size])  #
    un_x_d = tf.placeholder("float", [None, n_depths, input_size])  #
    un_y_phy = tf.placeholder("float", [None, n_steps])  #

    with tf.variable_scope("rnn1", reuse=True) as scope_sp:
        state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, unsup_inputs, dtype=tf.float32, scope=scope_sp)

    with tf.variable_scope("rnn2", reuse=True) as scope_sp1:
        state_series_xu_d, current_state_xu_d = tf.nn.dynamic_rnn(lstm_cell_depth, un_x_d, dtype=tf.float32,
                                                                  scope=scope_sp)

    keep_prob = 0.5
    pred_u = []
    for i in range(n_steps):
        tp2 = state_series_xu[:, i, :]
        pt2 = tf.matmul(tp2, w_fin) + b_fin
        # pt2 = tf.nn.dropout(pt2,keep_prob)
        pred_u.append(pt2)

    pred_u = tf.stack(pred_u, axis=1)  #
    pred_u = tf.reshape(pred_u, [-1, n_steps])  #

    pred_d_u = []
    for i in range(n_depths):
        tp1_d_u = state_series_xu_d[:, i, :]  #
        # print(tp1_d_u)
        pt_d_u = tf.matmul(tp1_d_u, w_fin_d) + b_fin_d  #
        # pt_d_u = tf.nn.dropout(pt_d_u, keep_prob)
        # print(pt_d_u)
        pred_d_u.append(pt_d_u)  #
    # print(np.array(pred_d_u).shape)  #
    # print(pred_d_u)
    pred_s_d_u = tf.reshape(pred_d_u, [-1, n_steps])

    y_s_phy_u = tf.reshape(un_y_phy, [-1, n_steps])  #

    # unpred_s_all = (pred_u+pred_s_d_u+y_s_phy_u)/3
    # unpred_s_all = pred_u+pred_s_d_u+y_s_phy_u
    unpred_s_all2 = w1 * pred_u + w2 * pred_s_d_u + w3 * y_s_phy_u + b1

    unsup_phys_data = tf.placeholder("float", [None, n_steps, phy_size])
    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    unsup_loss, a, b, c = calculate_ec_loss(unsup_inputs,
                                            unpred_s_all2,
                                            unsup_phys_data,
                                            depth_areas,
                                            n_depths,
                                            ec_threshold,
                                            combine_days=1)

    # d_loss = den_loss(unpred_s_all)
    cost = r_cost + elam * unsup_loss
    # cost = r_cost + elam * unsup_loss +dlam*d_loss

    tvars = tf.trainable_variables()

    # regularization_cost = l2_radio * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tvars ])
    # cost = r_cost + elam * unsup_loss+ regularization_cost
    # for i in tvars:
    #     print(i)
    grads = tf.gradients(cost, tvars)

    saver = tf.train.Saver(max_to_keep=5)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))




    depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
    n_depths = depth_areas.size

    print("run "+str(ii)+" times")



    x_full = np.load(os.path.join(args.data_path, 'processed_features'+str(ii)+'.npy'))
    yphy = np.load(os.path.join(args.data_path, 'labels_phy'+str(ii)+'.npy'))
    x_raw_full = np.load(os.path.join(args.data_path, 'features'+str(ii)+'.npy'))
    diag_full = np.load(os.path.join(args.data_path, 'diag.npy'))

    # ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Daily.Qe', 'Daily.Qh', 'Has.Black.Ice']
    phy_full = np.concatenate((x_raw_full[:,:,:-3],diag_full),axis=2)

    new_dates = np.load(os.path.join(args.data_path, 'dates.npy'), allow_pickle=True)

    # train_data = feather.read_dataframe(os.path.join(args.data_path, label_train))

    # tr_date = train_data.values[:,0]
    # tr_depth = train_data.values[:,1]
    # tr_temp = train_data.values[:,2]


    t_steps = x_raw_full.shape[1]#3549
    # print(t_steps)
    # m_tr = np.zeros([n_depths,t_steps])#50*3549
    # obs_tr = np.zeros([n_depths,t_steps])
    # k=0
    # #dd = 0
    # for i in range(new_dates.shape[0]):#3550
    #     if k>=tr_date.shape[0]:
    #         break
    #     # print(new_dates[i])
    #     # print(tr_date[k])
    #     while new_dates[i]==tr_date[k]:
    #         d = min(int(tr_depth[k]/0.5),n_depths-1)
    #         m_tr[d,i]=1
    #         obs_tr[d,i]=tr_temp[k]
    #         k+=1
    #         if k>=tr_date.shape[0]:
    #             break

    test_data = feather.read_dataframe(os.path.join(args.data_path, label_test))

    te_date = test_data.values[:,0]
    te_depth = test_data.values[:,1]
    te_temp = test_data.values[:,2]

    m_te = np.zeros([n_depths,t_steps])
    obs_te = np.zeros([n_depths,t_steps])
    k=0
    #dd = 0
    for i in range(new_dates.shape[0]):
        # print(k)
        # print(te_date.shape)
        if k>=te_date.shape[0]:
            break
        # print(new_dates[i])
        # print(te_date[k])
        while new_dates[i]==te_date[k]:
            d = min(int(te_depth[k]/0.5),n_depths-1)
    #        if m_te[d,i]==1:
    #            print(d,te_depth[k])
            m_te[d,i]=1
            obs_te[d,i]=te_temp[k]
            # print(te_temp[k])
            k+=1
            if k>=te_date.shape[0]:
                break
    # for i in range(0,obs_te.shape[1]):
    #     print(obs_te[0,i])

    x_train = np.zeros([n_depths*N_sec,n_steps,input_size])
    x_train_d = np.zeros([n_steps*N_sec,n_depths,input_size])
    y_train_phy = np.zeros([n_depths*N_sec,n_steps])
    y_train = np.zeros([n_depths*N_sec,n_steps])
    p_train = np.zeros([n_depths*N_sec,n_steps,phy_size])
    m_train = np.zeros([n_depths*N_sec,n_steps])
    y_test = np.zeros([n_depths*N_sec,n_steps])
    m_test = np.zeros([n_depths*N_sec,n_steps])

    x_full_d = x_full.transpose(1,0,2)

    for i in range(1,N_sec+1):
        x_train[(i-1)*n_depths:i*n_depths,:,:]=x_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
        y_train_phy[(i - 1) * n_depths:i * n_depths, :] = yphy[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
        # y_train[(i-1)*n_depths:i*n_depths,:]=obs_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        p_train[(i-1)*n_depths:i*n_depths,:,:]=phy_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
        # m_train[(i-1)*n_depths:i*n_depths,:]=m_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]

        y_test[(i-1)*n_depths:i*n_depths,:]=obs_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        m_test[(i-1)*n_depths:i*n_depths,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
        x_train_d[(i-1)*n_steps:i*n_steps,:,:] = x_full_d[int((i-1)*n_steps/2):int((i+1)*n_steps/2),:,:]



    # print(np.array(x_train).shape)
    # print(np.array(y_train).shape)
    # print(x_train[0,:,:])
    # print(x_train[50,:,:])
    # print(x_train[48,:,:])
    # print(y_train[0,:])
    # print(y_train[48,:])
    # print(y_train[50,:])
    # print(m_train[0,:])
    # print(m_train[48,:])
    # print(m_train[50,:])
    # print(y_test[0,:])
    # print(y_test[48,:])
    # print(y_test[450,:])
    ''' Test '''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # If using pretrained model, reload it now

        saver.restore(sess, os.path.join(save_path,  "trained_model2_+_"+str(ii)+"_"+str(epochs)+".ckpt"))
        rc_list = []
        # for epoch in range(epochs):
        #     loss = 0
        #     rc = 0
        #     ec = 0
        #     count = 0
        #     for i in range(1, N_sec+1):#12的时候mask为0
        #
        #         x_train_b = x_train[(i-1)*n_depths:i*n_depths,:,:]
        #         x_train_d_b = x_train_d[(i-1)*n_steps:i*n_steps,:,:]
        #         y_train_phy_b = y_train_phy[(i - 1) * n_depths:i * n_depths, :]
        #         y_train_b = y_train[(i-1)*n_depths:i*n_depths,:]
        #         m_train_b = m_train[(i-1)*n_depths:i*n_depths,:]
        #         x_f = x_train_b
        #         x_f_d = x_train_d_b
        #         y_f_phy = y_train_phy_b
        #         p_f = p_train[(i-1)*n_depths:i*n_depths,:,:]
        #         # print(np.array(x_train_b).shape)
        #         # print(np.array(x_train_d_b).shape)
        #         # print(np.array(y_train_phy_b).shape)
        #         # print(np.array(y_train_b).shape)
        #         # print(np.array(m_train_b).shape)
        #         # print(np.array(p_f).shape)
        #
        #         tt = np.sum(m_train_b, axis=0)
        #         aaaa=np.sum(tt,axis=0)
        #         if aaaa!=0:
        #
        #             _, loss_,rc_,ec_,aa,bb,cc,prd1 = sess.run(
        #                     [train_op, cost,r_cost,unsup_loss,a,b,c,pred_s_all2_old],
        #                     feed_dict = {
        #                             x: x_train_b,
        #                             x_d:x_train_d_b,
        #                             y_phy: y_train_phy_b,
        #                             y: y_train_b,
        #                             m: m_train_b,
        #                             unsup_inputs: x_f,
        #                             un_x_d:x_f_d,
        #                             un_y_phy:y_f_phy,
        #                             unsup_phys_data: p_f,
        #                             bt_sz: n_depths*N_sec
        #                 })
        #             print("batch " + str(i) + ", BatLoss= " + \
        #                   "{:.32f}".format(loss_) + ", Rc= " + \
        #                   "{:.32f}".format(rc_) + ", Ec= " + \
        #                   "{:.32f}".format(ec_))
        #             loss = loss +loss_
        #             rc = rc + rc_
        #             ec = ec + ec_
        #             count = count+1
        #         # with tf.Session() as sess:
        #         #     print(sess.run(aaaa))
        #     loss = loss/count
        #     rc = rc/count
        #     ec = ec/count
        #     rc_list.append(rc)
        #     if epoch%1==0:
        #         print("Step " + str(epoch) + ", BatLoss= " + \
        #           "{:.32f}".format(loss) + ", Rc= " + \
        #           "{:.32f}".format(rc) + ", Ec= " + \
        #           "{:.32f}".format(ec))
        if test == 1:
            loss_te = 0
            Ec_te = 0
            count_te = 0
            for i in range(1, N_sec + 1):
                x_train_b = x_train[(i - 1) * n_depths:i * n_depths, :, :]
                x_train_d_b = x_train_d[(i - 1) * n_steps:i * n_steps, :, :]
                y_train_phy_b = y_train_phy[(i - 1) * n_depths:i * n_depths, :]
                y_test_b = y_test[(i - 1) * n_depths:i * n_depths, :]
                m_test_b = m_test[(i - 1) * n_depths:i * n_depths, :]
                x_f = x_train_b
                x_f_d = x_train_d_b
                y_f_phy = y_train_phy_b
                p_f = p_train[(i - 1) * n_depths:i * n_depths, :, :]
                tt = np.sum(m_test_b, axis=0)
                aaaa2 = np.sum(tt, axis=0)
                # aaaa2 = tf.reduce_sum(tf.reshape(m_test_b, [-1, 1]))
                loss_te_, Ec_te_, prd = sess.run([r_cost, unsup_loss, pred_s_all2_old],
                                                 feed_dict={x: x_train_b, x_d: x_train_d_b, y_phy: y_train_phy_b,
                                                            y: y_test_b,
                                                            m: m_test_b, unsup_inputs: x_f, un_x_d: x_f_d,
                                                            un_y_phy: y_f_phy,
                                                            unsup_phys_data: p_f, bt_sz: n_depths * N_sec})
                if i == 1:
                    prd_t = prd
                if i > 1:
                    prd_t = np.vstack((prd_t, prd))
                if aaaa2 != 0:
                    # loss_te_,Ec_te_,prd = sess.run([r_cost,unsup_loss,pred], feed_dict = {x: x_train_b,x_d:x_train_d_b,y_phy: y_train_phy_b, y: y_test_b, m: m_test_b,unsup_inputs: x_f,un_x_d:x_f_d,un_y_phy:y_f_phy,unsup_phys_data: p_f, bt_sz: n_depths*N_sec})

                    loss_te = loss_te + loss_te_
                    Ec_te = Ec_te + Ec_te_
                    count_te = count_te + 1
                    print("batch " + str(i))
                    print("Loss_te " + "{:.4f}".format(loss_te_))
                    print("Ec= " + "{:.4f}".format(Ec_te_))
            loss_te = loss_te / count_te
            Ec_te = Ec_te / count_te
            print("Loss_te_mean " + "{:.4f}".format(loss_te))
            print("Ec_mean= " + "{:.4f}".format(Ec_te))
            # print(dd)
            # if args.save_path != '':
            # saver.save(sess, os.path.join(args.save_path, "trained_model2_+_"+str(ii)+"_"+str(epoch+1)+".ckpt"))

            # predict on test data, reshape to output file format, and save
            # loss_te,prd = sess.run([r_cost,pred], feed_dict = {x: x_train,x_d:x_train_d,y_phy: y_train_phy,y: y_test, m: m_test})
            # print("Loss_te " + "{:.4f}".format(loss_te))
            prd = prd_t
            prd_o = np.zeros([n_depths, n_steps + int((N_sec - 1) * n_steps / 2)])
            prd_o[:, :n_steps] = prd[0:n_depths, :, 0]
            for j in range(N_sec - 1):
                st_idx = n_steps - (int((j + 1) * n_steps / 2) - int(j * n_steps / 2))  # handle even or odd cases
                prd_o[:, n_steps + int(j * n_steps / 2):n_steps + int((j + 1) * n_steps / 2)] = prd[(j + 1) * n_depths:(
                                                                                                                                   j + 2) * n_depths,
                                                                                                st_idx:, 0]
            prd_o_ = np.transpose(prd_o)
            # np.savetxt(os.path.join(args.preds_path, "predict_model2_+_"+str(ii)+"_"+str(epochs+1)+".csv"), prd_o, delimiter=',')
            # np.savetxt(os.path.join(args.preds_path, "predict_pgdl55.csv"), prd_o_, delimiter=',')
        rc_np =  np.array(rc_list)
        return prd_o_
        # np.savetxt(os.path.join(args.preds_path, "train_loss2" + str(ii) + ".csv"), rc_np, delimiter=',')
