from improv.actor import Actor
import numpy as np
import logging
from adaptive_latents import proSVD, CenteringTransformer, KernelSmoother
from adaptive_latents.stim_designer import StimDesigner
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DimRed(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.centerer = CenteringTransformer()
        self.smoother = KernelSmoother()
        self.pro = proSVD(k=10)

        self.stim_designer = StimDesigner()

    def setup(self):
        pass

    def runStep(self):
        data = self.links['q_in'].get(timeout=.001)

        data = self.centerer.partial_fit_transform(data)
        data = self.smoother.partial_fit_transform(data)
        data = self.pro.partial_fit_transform(data)

        if self.pro.is_initialized:
            v = np.zeros([10,1])
            v[0] = 1
            logger.info(f'u_dimension={self.pro.Q[0]}')
            logger.info(f'u_dimension={self.pro.Q.shape}')
            stim = self.stim_designer.design_stim(v=v, u_dimension=self.pro.Q.shape[0], u_to_s_function= lambda u: self.pro.Q.T @ u)
            logger.info(stim)

    def stop(self):
        pass
