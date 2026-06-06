from improv.actor import Actor
from queue import Empty
import numpy as np
import time
import logging
from adaptive_latents import proSVD, CenteringEstimator, KernelSmoother, ArrayWithTime
from adaptive_latents.stim_designer import StimDesigner, OptimizationMethod
from adaptive_latents.stim_regressor import StimRegressor, StimEvent, StreamingKalmanFilter, BaseMultiKernelRegressor
from adaptive_latents.estimator import StreamingEstimator
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("jax").setLevel(logging.ERROR)

LOG_LEVEL = 0


class ImprovStimDesigner(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.frame_number = None
        self.coords = None

        self.centerer = CenteringEstimator(log_level=LOG_LEVEL)
        self.smoother = KernelSmoother(log_level=LOG_LEVEL)
        self.pro = proSVD(k=10, log_level=LOG_LEVEL)

        self.stim_designer = StimDesigner(
            should_log=LOG_LEVEL,
            optimization_method=OptimizationMethod.HOMOGENOUS
        )
        self.stim_regressor = StimRegressor(
            autoreg=StreamingKalmanFilter(
                steps_between_refits=5,
                log_level=LOG_LEVEL,
                max_history_length=1000,
            ),
            stim_reg=BaseMultiKernelRegressor(maxlen=100, should_log=LOG_LEVEL),
            stim_delay=0,
            log_level=LOG_LEVEL,
            error_on_missed_stim=True,
        )

    def runStep(self):
        # if not hasattr(self, 'debugpy'): import debugpy; debugpy.listen(5678); debugpy.debug_this_thread(); debugpy.wait_for_client(); self.debugpy = debugpy
        # self.debugpy.breakpoint()

        # if not hasattr(self, 'pudb_remote'): from pudb import remote as pudb_remote; self.pudb_remote = pudb_remote;
        # self.pudb_remote.set_trace(host='127.0.0.1', port=6899)  # connect with `telnet 127.0.0.1 6899`

        start_time = time.time()
        try:
            ids = self.q_in.get(timeout=.001)
        except Empty:
            pass
        else:
            self.handle_new_data(
                C=self.client.get(ids[2]),
                coords=self.client.get(ids[0]),
                frame_number=ids[3],
            )

        if 'necessary' and False:
            try:
                stim_frame = self.links['stim_happened'].get(timeout=.001)
                pass
            except Empty:
                pass
            else:
                'retroactively place time'

        elapsed_time = time.time() - start_time
        # logger.info(f'runStep time: {elapsed_time*1000:.1f} ms')

    def handle_new_data(self, C: np.ndarray, coords, frame_number: int):
        self.coords = coords
        old_frame_number = self.frame_number
        self.frame_number = frame_number

        if old_frame_number is None:
            old_frame_number = self.frame_number - C.shape[1]

        n_new_frames = frame_number - old_frame_number

        data = C[:, C.shape[1] - n_new_frames:].T

        data = ArrayWithTime(data, t=np.arange(old_frame_number+1,self.frame_number+1))

        for row in data:
            reshaped_row = row[None, :]
            reshaped_row.t = row.t[0]
            self.process_new_row(reshaped_row)



    def process_new_row(self, data: ArrayWithTime):
        assert data.shape[0] == 1 and len(data.shape) == 2

        # data = self.centerer.step(data)
        # data = self.smoother.partial_fit_transform(data)
        if self.pro.Q is not None and data.shape[1] > self.pro.Q.shape[0]:
            n_new_channels = data.shape[1] - self.pro.Q.shape[0]
            self.pro.add_new_input_channels(n_new_channels)
            logger.info(f'new C shape: ({data.shape[1]}xN)')
            if self.stim_regressor.stim_reg.input_histories is not None:
                old_u_history = self.stim_regressor.stim_reg.input_histories[1]
                self.stim_regressor.stim_reg.input_histories[1] = np.hstack((old_u_history, np.zeros((old_u_history.shape[0], n_new_channels))))
        data = self.pro.step(data)

        if self.pro.is_initialized:
            v = np.zeros([10, 1])
            v[0] = 1
            R_U, _, _ = np.linalg.svd(self.pro.R)  # this is fast, R is small
            stim = self.stim_designer.design_stim(v=R_U @ v, u_dimension=self.pro.Q.shape[0],
                                                  u_to_s_function=lambda u: self.pro.Q.T @ u)


            if self.frame_number > 20 and self.frame_number % 20 == 0:
                id = self.client.put(stim)
                self.links['stim_vector_out'].put(id)

                self.handle_stim(
                    u=stim,
                    delivery_time=self.frame_number,
                )

                if self.stim_regressor.stim_reg.output_history is not None:
                    logger.info(f'jdg: output history length: {np.all(self.stim_regressor.stim_reg.output_history != 0, axis=1).sum()})')
            self.stim_regressor.step(data)




    def handle_stim(self, u, delivery_time=None):
        dt = self.stim_regressor.dt
        stim_delay = 0

        self.stim_regressor.add_event(
            StimEvent(
                u=u,
                delivery_time=delivery_time,
                no_fit_interval=(delivery_time, delivery_time + stim_delay),
                difference_interval=(delivery_time + stim_delay - dt, delivery_time + stim_delay),
                no_observe_interval=None,
                eps=dt / 8,
                error_on_missed=self.stim_regressor.error_on_missed_stim,
            )
        )

        # we can figure this out if it becomes necessary
        # if 'we need to update retroactively' and False:
        #     updated_delivery_time = ...
        #     self.stim_regressor.ignore_data_events[-1].difference_interval = (updated_delivery_time - dt, updated_delivery_time)

    def stop(self):
        with (open('output/design_stim.pickle', 'wb') as f):
            d = {}
            for k, v in self.__dict__.items():
                # TODO: do this better
                if isinstance(v, StreamingEstimator) or isinstance(v, np.ndarray) or isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    d[k] = v
            pickle.dump(d, f)