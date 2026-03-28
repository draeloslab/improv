from improv.actor import Actor
from queue import Empty
import numpy as np
import time
import logging
from adaptive_latents import proSVD, CenteringEstimator, KernelSmoother
from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents.stim_regressor import StimRegressor, StimEvent
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


logging.getLogger("jax").setLevel(logging.ERROR)


class ImprovStimDesigner(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = None
        self.coords = None

        self.centerer = CenteringEstimator(log_level=0)
        self.smoother = KernelSmoother(log_level=0)
        self.pro = proSVD(k=10, log_level=0)

        self.stim_designer = StimDesigner(should_log=False)
        self.stim_regressor = StimRegressor()

    def setup(self):
        pass

    def handle_stim(self, u, delivery_time=None):
        dt = self.stim_regressor.dt

        self.stim_regressor.add_event(
            StimEvent(
                u=u,
                delivery_time=delivery_time,
                no_fit_interval = (delivery_time, delivery_time),
                difference_interval = (delivery_time-dt, None),
                no_observe_interval=None,
                eps=0
            )
        )

        if 'we need to update':
            updated_delivery_time = ...
            self.stim_regressor.ignore_data_events[-1].difference_interval = (updated_delivery_time-dt, updated_delivery_time)

    def runStep(self):
        start_time = time.time()
        try:
            ids = self.links['q_in'].get(timeout=.001)
        except Empty:
            return

        C = self.client.get(ids[0])
        self.coords = self.client.get(ids[1])

        columns_seen = self.C.shape[1] if self.C is not None else 0
        data = C[:,columns_seen:].T


        # data = self.centerer.partial_fit_transform(data)
        # data = self.smoother.partial_fit_transform(data)
        if self.pro.Q is not None and data.shape[1] > self.pro.Q.shape[0]:
            self.pro.add_new_input_channels(data.shape[1] - self.pro.Q.shape[0])
            logger.info(f'jdg: new C shape: {C.shape}')
        data = self.pro.step(data)

        if self.pro.is_initialized:
            v = np.zeros([10,1])
            v[0] = 1
            stim = self.stim_designer.design_stim(v=v, u_dimension=self.pro.Q.shape[0], u_to_s_function= lambda u: self.pro.Q.T @ u)
            # logger.info(stim)

        self.C = C
        elapsed_time = time.time() - start_time
        logger.info(f'jdg: runStep time: {elapsed_time:.4f} s')

    def stop(self):
        pass
