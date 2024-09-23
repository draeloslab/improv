import time
import numpy as np
import random
import zmq
from improv.actor import Actor
from queue import Empty
from scipy.stats import norm
import random

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0
        self.stop_sending = False

        self.seed = 42 #1337 #81 #1337 #7419
        np.random.seed(self.seed)

        self.prepared_frame = None
        self.random_flag = False
        
        self.stimuli = np.load(stimuli, allow_pickle=True)
        np.save('output/generated_stimuli.npy', self.stimuli)

        self.initial = True
        self.newN = False
        self.circ_size = 10

        self.stim_choice = []
        self.GP_stimuli = []
        self.GP_stimuli_init = []
        self.stim_sets = []
        for i,s in enumerate(self.stimuli):
            print(i,s)
            self.stim_choice.append(s.shape[0])
    
            indices = np.arange(s.shape[0])
            random.shuffle(indices)        

            self.GP_stimuli_init.append(np.arange(s.shape[0])[indices])
            self.GP_stimuli.append(np.arange(s.shape[0]))
            self.stim_sets.append(s[indices].tolist())
        self.stim_choice = np.array(self.stim_choice)

        snum = self.stim_choice[0]
        print('Number of possible stimuli is :', snum)
        self.grid_choice = np.arange(snum) #**2)
        np.random.shuffle(self.grid_choice)
        self.grid_choice = np.reshape(self.grid_choice, (snum,)) #snum))
        self.grid_ind = np.arange(snum) #**2)

        self.initial_angles = np.linspace(1, 10, num=10) # np.linspace(0,360,endpoint=False, num=8)
        self.initial_vel = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
        random.shuffle(self.initial_angles)
        random.shuffle(self.initial_vel)
        self.which_angle = 0
        self.all_angles = self.stim_sets[0]
        random.shuffle(self.all_angles)

        ### Optimizer
        maxS = self.stim_choice #np.array([l[-1] for l in self.stim_choice])
        gamma = maxS #(1 / maxS) / 2
        print(gamma)
        var = 0.5 #1e-1
        nu = 1 #0.5 #1e-1
        eta = 5e-2 #8e-2 #-1e-2
        d = 2

        gp_copy = self.GP_stimuli.copy()
        xs = np.meshgrid(*gp_copy) #,x3,x4])
        x_star = np.empty(xs[0].shape + (d,))
        for i in range(d):
            x_star[...,i] = xs[i]

        self.x_star = x_star.reshape(-1, d)      #shape (a,d) where a is all possible test points
        print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        self.optim = Optimizer(gamma[:d], var, nu, eta, self.x_star)
        init_T = 4
        self.X0 = np.zeros((d, init_T))
        self.X = self.X0.copy()
        self.y0 = None

        self.nID = None
        self.conf = None
        self.maxT = 20

        ## random sampling for initialization
        self.initial_length = 10 #16*2 #16*3

        self.optimized_n = []

        self.saved_GP_est = []
        self.saved_GP_unc = []

        xs = np.meshgrid(*self.stimuli) #,x3,x4])
        x_star = np.empty(xs[0].shape + (d,))
        for i in range(d):
            x_star[...,i] = xs[i]

        self.stim_star = x_star.reshape(-1, d)

        print(self.stim_choice, self.GP_stimuli)

        self.stopping_list = []
        self.peak_list = []
        self.goback_neurons = []
        self.optim_f_list = []

        if self.random_flag:
            self.stimuli = self.stimuli.copy()[:, ::2]

            snum = int(self.stim_choice[0] / 2)
            print('Number of angles during grid phase since random is :', snum)
            self.grid_choice = np.arange(snum**2)
            np.random.shuffle(self.grid_choice)
            self.grid_choice = np.reshape(self.grid_choice, (snum,snum))
            print(self.grid_choice)
            self.grid_ind = np.arange(snum**2)

        # context = zmq.Context()
        
        # print('Starting setup')
        # self._socket = context.socket(zmq.PUB)
        # send_IP =  self.ip
        # send_port = self.port
        # self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        # self.stimulus_topic = 'stim'
        # print('Done setup VisStim')


    def setup(self):
        logger.info('Someone is running my setup function')
        context = zmq.Context()
        
        print('Starting setup')
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'
        print('Done setup VisStim')

        self.timer = time.time()
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []

    def stop(self):
        '''Triggered at Run
        '''
        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        # print(self.stopping_list)
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        # print(self.peak_list)
        np.save('output/peak_list.npy', np.array(self.peak_list))
        # print(self.optim_f_list)
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        
    def runStep(self):
        ### Get data from analysis actor
        try:
            ids = self.q_in.get(timeout=0.0001)

            # X, Y, stim, _ = self.client.get(ids)
            X = self.client.get(ids[0])
            Y = self.client.get(ids[1])
            stim = self.client.get(ids[2])

            tmpX = np.squeeze(np.array(X)).T
            # logger.info(f'{tmpX.shape}, {len(Y)}----------------------------------------------------')
            sh = len(tmpX.shape)
            if sh > 1:
                self.X = tmpX.copy()
                if tmpX.shape[1] > 4:
                    self.X = tmpX[:, -tmpX.shape[1]:]
                # print('self.X DIRECT from analysis is ', X, 'and self.X is ', self.X[:,-1])
            # print(self.X)

            try:
                b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
                for i,j in enumerate(Y):
                    b[i][:len(j)] = j
                self.y0 = b.T
                # logger.info('X, Y shapes: {}, {}'.format(self.X.shape, self.y0.shape))
            except:
                pass
            

        except Empty as e:
            pass
        except Exception as e:
            print('Error in stimulus_spots get: {}'.format(e))



        ### initial run ignores signals and just sends 8 basic stimuli
        if self.stop_sending:
            pass
            
        elif self.initial:
            # pass
            if self.prepared_frame is None:
                self.prepared_frame = self.initial_frame()
                # logger.info('got circ size {}'.format(self.circ_size))
                # logger.info(self.prepared_frame)
                # self.prepared_frame.pop('load')
            if (time.time() - self.timer) >= self.total_stim_time:
                # logger.info(self.prepared_frame)
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None


        ### once initial done, or we move on, initial GP with next neuron
        elif self.newN:
            # # ## doing random stims
            if self.random_flag:
                
                if self.prepared_frame is None:
                    self.prepared_frame = self.random_frame()
                    # self.prepared_frame.pop('load')
                if (time.time() - self.timer) >= self.total_stim_time:
                        # self.random_frame()
                    self.send_frame(self.prepared_frame)
                    self.prepared_frame = None

            else:
                # print(self.optimized_n, set(self.optimized_n))
                nonopt = np.array(list(set(np.arange(self.y0.shape[0]))-set(self.optimized_n)))
                logger.info('nonopt is {}, number of neurons '.format(nonopt,self.y0.shape[0]))
                if len(nonopt) >= 1 or len(self.goback_neurons)>=1:
                    if len(nonopt) >= 1:
                        self.nID = nonopt[np.argmax(np.mean(self.y0[nonopt,:], axis=1))]
                        logger.info('selecting most responsive neuron: {}'.format(self.nID))
                        self.optimized_n.append(self.nID)
                        self.saved_GP_est = []
                        self.saved_GP_unc = []
                    elif len(self.goback_neurons)>=1:
                        self.nID = self.goback_neurons.pop(0)
                        print('Trying again with neuron', self.nID)
                        self.optimized_n.append(self.nID)
                    
                    print(self.y0.shape, self.X.shape, self.X0.shape)
                    if self.X.shape[1] < self.y0.shape[1]:
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, -self.X.shape[1]:].T)
                    elif self.y0.shape[1] < self.maxT:
                        self.optim.initialize_GP(self.X[:, -self.y0.shape[1]:].T, self.y0[self.nID, -self.y0.shape[1]:].T)
                    else:
                        # self.optim.initialize_GP(self.X[:, -self.maxT:].T, self.y0[self.nID, -self.maxT:].T)
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, :].T)
                    # print('known average sigma, ', np.mean(self.optim.sigma))
                    # self.optim.initialize_GP(self.X0[:, :3], self.y0[self.nID, :3])
                    self.test_count = 0
                    self.newN = False
                    self.stopping = np.zeros(self.maxT)

                    curr_unc = np.diagonal(self.optim.sigma.reshape((60,60))).reshape((10,6))
                    curr_est = self.optim.f.reshape((10,6))
                    self.saved_GP_unc.append(curr_unc)
                    self.saved_GP_est.append(curr_est)

                    ids = []
                    # print('--------------- nID', self.nID)
                    # ids.append(self.client.put(self.nID, 'nID'))
                    ids.append(self.nID)
                    ids.append(self.client.put(curr_est)) #, 'est'))
                    ids.append(self.client.put(curr_unc)) #, 'unc'))
                    # ids.append(self.client.put(self.conf, 'conf'))
                    self.q_out.put(ids)
                
                # else:
                #     self.initial = True
                #     print('----------------- done with this plane, moving to next')
                #     self.send_move(10)

        ### update GP, suggest next stim
        else:
            
            if self.prepared_frame is None:
                X = np.zeros(2)
                # print('self.X from analysis is ', self.X[:,-1])
                # print('going back further', self.X)
                # print('GP_stimuli', self.GP_stimuli[0])
                # try:
                X[0] = self.GP_stimuli[0][int(self.X[0,-1])]
                X[1] = self.GP_stimuli[1][int(self.X[1,-1])]
                # X[2] = self.GP_stimuli[2][int(self.X[2,-1])]
                logger.info('optim {} , update GP with {}, {}'.format( self.nID, X, self.y0[self.nID, -1]))
                self.optim.update_GP(np.squeeze(X), self.y0[self.nID,-1])

                curr_unc = np.diagonal(self.optim.sigma.reshape((60,60))).reshape((10,6))
                curr_est = self.optim.f.reshape((10,6))
                self.saved_GP_unc.append(curr_unc)
                self.saved_GP_est.append(curr_est)
                # except:
                #     pass
                ids = []
                # print('--------------- nID', self.nID)
                # ids.append(self.client.put(self.nID, 'nID'))
                ids.append(self.nID)
                ids.append(self.client.put(curr_est)) #, 'est'))
                ids.append(self.client.put(curr_unc)) #, 'unc'))
                # ids.append(self.client.put(self.conf, 'conf'))
                self.q_out.put(ids)

                stopCrit = self.optim.stopping()
                logger.info('----------- stopCrit: {}'.format(stopCrit))
                self.stopping[self.test_count] = stopCrit
                self.test_count += 1

                
                if stopCrit < 3.0e-4: #6.0e-4: #8e-2: #0.37/2.05
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    logger.info('Satisfied with this neuron, moving to next. Est peak: {}'.format(peak))
                    # self.nID += 1
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                    np.save('output/saved_GP_est_'+str(self.nID)+'.npy', np.array(self.saved_GP_est))
                    np.save('output/saved_GP_unc_'+str(self.nID)+'.npy', np.array(self.saved_GP_unc))

                    # if len(self.optim_f_list) >= 500:
                    #     print('----------------- done with this plane, moving to next')
                    #     self.send_move(10)
                    
                elif self.test_count >= self.maxT:
                    logger.info('exceeded test count')
                    self.goback_neurons.append(self.nID)
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                else:
                    ind, xt_1 = self.optim.max_acq()
                    logger.info('suggest next stim: {}, {}, {}'.format(ind, xt_1, xt_1.T[...,None].shape))
                    self.prepared_frame = self.create_chosen_stim(ind)
 
            if (time.time() - self.timer) >= self.total_stim_time:
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None

    def send_frame(self, stim):

        circ_size = self.circ_size

        if stim is not None:
            text = {'texture_size': 1024, 
                    'circle_center': (35,35),
                    'circle_radius': circ_size,
                    'texture_name':'gray_circle',
                    'bg_intensity': 200, #255,#0,
                    'fg_intensity': 50, #0,#255,
                    }

            stimulus = {'stimulus': stim, 'texture': text}
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            logger.info('Number of stimuli requested: {}'.format(self.displayed_stim_num))
            self.displayed_stim_num += 1
        else:
            logger.error('Tried to send a None frame')

    def send_move(self, z):
        self._socket.send_string('move', zmq.SNDMORE)
        self._socket.send_pyobj(z)
        logger.info('sent move command')

    def create_chosen_stim(self, ind):
        xt = self.stim_star[ind]
        angle = xt[0]
        angle2 = xt[1]
        stim = self.create_frame(angle, angle2)
        return stim

    def create_frame(self, circle_size, vel, angle=0):
        ### Static or common stimulus params are set here
        # angle = 270
        # vel = 0.04

        self.circ_size = circle_size

        stat_t = 10 #0
        stim_t = stat_t + 25
        self.total_stim_time = stim_t
    
        stim = {
                'stim_name': 'circle_radius',
                'angle': angle,
                'velocity': vel,
                'stationary_time': stat_t,
                'duration': stim_t,
                'hold_after': float(stim_t),
                    }

        self.timer = time.time()
        return stim

    def initial_frame(self):
        if self.which_angle%6 == 0:
            random.shuffle(self.initial_angles)
            random.shuffle(self.initial_vel)
        angle = self.initial_angles[self.which_angle%8] #self.stim_sets[0][self.which_angle%len(self.stim_sets[0])]
        vel = self.initial_vel[self.which_angle%6]

        self.which_angle += 1
        if self.which_angle >= self.initial_length: 
            self.initial = False
            self.stop_sending = False
            self.newN = True
            self.which_angle = 0
            logger.info('Done with initial frames, starting random set')
        
        stim = self.create_frame(angle, vel)
        self.timer = time.time()
        self.circ_size = angle
        return stim


    def random_frame(self):
        ## grid choice
        snum = int(self.stim_choice[0] / 2)
        grid = np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(snum**2)])[0] #self.which_angle%24 #self.which_angle%24 #np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(36*36)])[0]
        angle = self.stimuli[0][grid[0]] #self.all_angles[grid] #self.stimuli[0][grid[0]]
        angle2 = self.stimuli[0][grid[1]]

        stim = self.create_frame(angle, angle2)
        self.timer = time.time()
        return stim


class Optimizer():
    def __init__(self, gamma, var, nu, eta, x_star):
        self.gamma = gamma
        self.variance = var
        self.nu = nu
        self.eta = eta
        self.x_star = x_star        

        self.d = self.x_star.shape[1]

        self.f = None
        self.sigma = None       ## Note: this is actually sigma squared
        self.X_t = None
        self.K_t = None
        self.k_star = None
        self.y = None
        self.A = None

        self.t = 0

    def initialize_GP(self, X, y):
        ## X is a matrix (T,d) of initial T measurements we have results for

        self.X_t = X
        self.y = y

        T = self.X_t.shape[0]
        a = self.x_star.shape[0]

        self.test_count = np.zeros(a)

        self.K_t = kernel(self.X_t, self.X_t, self.variance, self.gamma)
        self.k_star = kernel(self.X_t, self.x_star, self.variance, self.gamma)

        self.A = np.linalg.inv(self.K_t + self.eta**2 * np.eye(T))
        self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(a) - self.k_star.T @ self.A @ self.k_star
        ### TODO: rewrite sigma computation to be every a not matrix mult
        # self.sigma = np.diagonal(self.sigma)

        self.t = T

    def update_obs(self, x, y):
        self.y_t1 = np.array([y])
        self.x_t1 = x[None,...]
       
    def update_GP(self, x, y):
        self.update_obs(x, y)

        ## Can't do internally due to out of memory / invalid array errors from numpy
        self.k_t, self.u, self.phi, f_upd, sigma_upd = update_GP_ext(self.X_t, self.x_t1, self.A, self.x_star, self.eta, self.y, self.y_t1, self.k_star, self.variance, self.gamma)
        self.f = self.f + f_upd
        # self.sigma = self.sigma + np.diagonal(sigma_upd)
        # self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(self.x_star.shape[0]) - self.k_star.T @ self.A @ self.k_star
        # self.sigma = np.diagonal(sigma)

        self.iterate_vars()

    def iterate_vars(self):
        self.y = np.append(self.y, self.y_t1)
        self.X_t = np.append(self.X_t, self.x_t1, axis=0)
        self.k_star = np.append(self.k_star, kernel(self.x_t1, self.x_star, self.variance, self.gamma), axis=0)

        ## update for A
        self.A = self.A + self.phi * np.outer(self.u, self.u)
        self.A = np.vstack((self.A, -self.phi*self.u.T))
        right = np.append(-self.phi*self.u, self.phi)
        self.A = np.column_stack((self.A, right))

        self.t += 1

    def max_acq(self):
        test_pt = np.argmax(self.ucb())
        
        if self.test_count[test_pt] > 5:
            test_pt = np.random.choice(np.arange(self.x_star.shape[0]))
            print('choosing random stim instead')
        self.test_count[test_pt] += 1

        return test_pt, self.x_star[test_pt]

    def ucb(self):
        tau = self.d * np.log(self.t + 1e-16)
        # import pdb; pdb.set_trace()
        sig = self.sigma
        if np.any(sig < 0):
            sig = np.clip(sig, 0, np.max(sig))
        fcn = self.f + np.sqrt(self.nu * tau) * np.sqrt(np.diagonal(sig))
        return fcn

    def stopping(self):
        val = self.f - np.max(self.f) - 1e-4
        # PI = np.max(norm.cdf((val) / (np.diagonal(self.sigma))))
        # using expected improvement
        sig = np.diagonal(self.sigma)
        EI = np.max(val * norm.cdf(val / sig) + sig * norm.pdf(val))
        return EI


def kernel(x, x_j, variance, gamma):
    ## x shape: (T, d) (# tests, # dimensions)
    K = np.zeros((x.shape[0], x_j.shape[0]))
    # period = 24 ##FIXME

    for i in range(x.shape[0]):
        # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
        for j in range(x_j.shape[0]):
            ## first dimension is direction
            # dist = np.abs(x[i,0] - x_j[j,0])
            # # print(dist)
            # # if dist > 12:
            # #     dist = 24 - dist
            # # print(dist)
            # K[i,j] = np.exp(-gamma[0]*((dist)**2))
            # K[i,j] *= variance * np.exp(-gamma[1:].dot((x[i,1:]-x_j[j,1:])**2))

            ## binocular
            # dist1 = np.sin(np.pi * np.abs(x[i,0] - x_j[j,0]) / period)
            # dist2 = np.sin(np.pi * np.abs(x[i,1] - x_j[j,1]) / period)

            dist1 = np.abs(x[i,0] - x_j[j,0])
            dist2 = np.abs(x[i,1] - x_j[j,1])

            K[i,j] = np.exp(-gamma[0]*(dist1**2))
            K[i,j] *= variance * np.exp(-gamma[1]*(dist2**2))

            
    return K

def update_GP_ext(X_t, x_t1, A, x_star, eta, y, y_t1, k_star, variance, gamma):

    k_t = kernel(X_t, x_t1, variance, gamma)
    u = A @ k_t
    k_t1 = kernel(x_t1, x_t1, variance, gamma)
    k_star_t1 = kernel(x_t1, x_star, variance, gamma)
    phi = np.linalg.inv(k_t1 + eta**2 - k_t.T.dot(u))
    kuk = k_star.T @ u - k_star_t1.T
    f = np.squeeze(phi * kuk * (y.dot(u) - y_t1))
    sigma = phi * (kuk**2)
    # import pdb; pdb.set_trace()

    return k_t, u, phi, f, sigma 

