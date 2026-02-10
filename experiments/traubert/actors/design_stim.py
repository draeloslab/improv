# estimator.py

import contextlib
import typing
from abc import ABC, abstractmethod

import numpy as np
from frozendict import frozendict
from tqdm.auto import tqdm
import types

class DataSource(ABC):
    @abstractmethod
    def next_sample_time(self) -> float:
        pass

    @abstractmethod
    def current_sample_time(self) -> float:
        pass

    @abstractmethod
    def __next__(self):
        "all data sources should be iterable"
        pass


class GeneratorDataSource(DataSource):
    def __init__(self, source, dt=1):
        if isinstance(source, types.GeneratorType):
            generator = source
        else:
            generator = iter(source)
        self.generator = enumerate(generator)
        self.next_sample = next(self.generator)
        self._current_time = None
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_sample[0] == float('inf'):
            raise StopIteration()

        this_sample = self.next_sample
        try:
            self.next_sample = next(self.generator)
        except StopIteration:
            self.next_sample = (float('inf'), None)

        self._current_time = this_sample[0]
        return ArrayWithTime(this_sample[1], t=self._current_time * self.dt)

    def next_sample_time(self):
        return self.next_sample[0]

    def current_sample_time(self):
        return self._current_time


class NumpyTimedDataSource(DataSource):
    def __init__(self, source, timepoints=None):
        self.a = source
        self.t = timepoints if timepoints is not None else np.arange(self.a.shape[0])
        assert len(self.t) == len(self.a)

        self.index = 0

    def __next__(self):
        try:
            d = self.a[self.index]
        except IndexError:
            raise StopIteration()

        d = ArrayWithTime(d.copy(), t=self.t[self.index])
        self.index += 1
        return d

    def next_sample_time(self):
        if self.index >= len(self.t):
            return float('inf')
        return self.t[self.index]

    def current_sample_time(self):
        if self.index == 0:
            return None
        return self.t[self.index-1]


class ArrayWithTime(np.ndarray):
    "The idea is to subclass here, but it seems pretty involved."
    # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    # https://stackoverflow.com/a/51955094
    def __new__(cls, input_array, t):
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        if hasattr(obj, 't'):
            self.t = obj.t

    def __reduce__(self):
        return self.__class__, (np.asarray(self), np.asarray(self.t))

    # def __getstate__(self):
    #     return super().__getstate__(), self.t
    #
    # def __setstate__(self, state):
    #     print(state)
    #     self.t = state[1]
    #     self.__setstate__(state[0])

    def __iter__(self):
        if hasattr(self.t, "__len__") and len(self.t) > 1 and len(self.t) == self.shape[0]:
            return NumpyTimedDataSource(self, self.t)
        else:
            return super().__iter__()

    def slice(self, *args, all_axes=False):
        # TODO: decide between 3.10 and 3.11 syntax; both pass tests
        if not all_axes:
            # return ArrayWithTime(self[*args], self.t[*args])
            return ArrayWithTime(self[args], self.t[args])
        elif all_axes:
            # return ArrayWithTime(self[*args], self.t[args[0]])
            return ArrayWithTime(self[args], self.t[args[0]])
        else:
            raise ValueError()

    def slice_by_time(self, *args, all_axes=False):
        def convert_from_time_to_indices(x):
            if isinstance(x, slice):
                assert x.step is None
                start, stop = x.start, x.stop
                if start is None:
                    start = self.t.min()
                if stop is None:
                    stop = self.t.max()
                if stop < start:
                    warnings.warn('stop greater than start; remember that time can be negative in slices')
                start = np.searchsorted(self.t, start, side='left')
                stop = np.searchsorted(self.t, stop, side='right')
                return slice(start, stop)
            elif x is ...:
                return x
            else:
                return self.time_to_sample(x)

        if len(args):
            if all_axes:
                args = (convert_from_time_to_indices(args[0]),) + args[1:]
            else:
                args = tuple(convert_from_time_to_indices(x) for x in args)

        return self.slice(*args, all_axes=all_axes)

    def as_array(self):
        return np.array(self)

    def time_to_sample(self, time):
        return np.searchsorted(self.t, time)

    @staticmethod
    def align_indices(a, b, complement=False):
        assert len(a.t) > 0 and len(b.t) > 0, 'neither of the arrays should be empty'
        # there's a faster way to do this with np.searchsorted
        a_t = np.array(a.t)
        b_t = np.array(b.t)
        a: ArrayWithTime
        assert (a_t[1:] - a_t[:-1] > 0).all()
        assert (b_t[1:] - b_t[:-1] > 0).all()
        idx_a = 0
        idx_b = 0
        a_indices = []
        b_indices = []

        while idx_a < len(a) and idx_b < len(b):
            d = a_t[idx_a] - b_t[idx_b]
            if np.isclose(0,d):
                a_indices.append(idx_a)
                b_indices.append(idx_b)
                idx_b += 1
                idx_a += 1
            elif d > 0:
                idx_b += 1
            else:
                idx_a += 1
        a_indices = np.array(a_indices)
        b_indices = np.array(b_indices)
        if complement:
            a_indices = np.setdiff1d(np.arange(len(a)), a_indices)
            b_indices = np.setdiff1d(np.arange(len(b)), b_indices)
        return ArrayWithTime(a[a_indices], a_t[a_indices]), ArrayWithTime(b[b_indices], b_t[b_indices])

    @staticmethod
    def subtract_aligned_indices(a, b):
        a, b = ArrayWithTime.align_indices(a, b)
        return ArrayWithTime(a - b, a.t)

    @property
    def dt(self):
        dts = np.diff(self.t)
        dt = np.median(dts)
        assert np.ptp(dts)/dt < 0.05
        return dt

    @staticmethod
    def from_list(input_list, squeeze_type='none', drop_early_nans=False, reshape_mid_nans=True):
        if len(input_list) and not hasattr(input_list[-1], 't'):
            warnings.warn("guessing t for input list")
            input_list = [ArrayWithTime(x, i) for i, x in enumerate(input_list)]

        if drop_early_nans:
            i = 0
            while i < len(input_list) and not np.isfinite(input_list[i]).all():
                i += 1
            input_list = input_list[i:]

        if reshape_mid_nans:
            for i in range(len(input_list)):
                hit = False
                if not np.isfinite(input_list[i]).any() and len(np.array(input_list[i]).shape) and np.array(input_list[i]).shape[-1] != np.array(input_list[0]).shape[-1]:
                    hit = True
                    input_list[i] = input_list[i][..., :np.shape(input_list[0])[-1]]
                    assert input_list[i].shape == np.array(input_list[0]).shape
                if hit:
                    warnings.warn('truncated an all-nan in the middle of a run')

        t = np.array([x.t for x in input_list])
        if squeeze_type == 'none' or squeeze_type is None:
            input_array = np.array(input_list)
        elif squeeze_type == 'to_2d':
            input_array = np.squeeze(input_list)
            if len(input_array.shape) == 1:
                input_array = input_array[:, None]
            elif len(input_array.shape) == 3:
                # warnings.warn("squeezing 3d array to 2d, this is unusual")
                input_array = input_array.reshape([-1, input_array.shape[-1]])
            assert len(input_array.shape) == 2
        elif squeeze_type == 'squeeze':
            input_array = np.squeeze(input_list)
        else:
            raise ValueError()

        return ArrayWithTime(input_array=input_array, t=t)

    @staticmethod
    def from_NTDS(ds: NumpyTimedDataSource):
        return ArrayWithTime(np.squeeze(ds.a, axis=1), ds.t)

    @staticmethod
    def from_transformed_data(new_data, old_data):
        # refers to the outputs of a transformer
        new_data = np.array(new_data)
        if hasattr(old_data, 't'):
            return ArrayWithTime(new_data, old_data.t)
        else:
            return new_data

    @staticmethod
    def from_nwb_timeseries(timeseries):
        return ArrayWithTime(timeseries.data[:], timeseries.timestamps[:])


    @staticmethod
    def from_notime(a):
        return ArrayWithTime(a, np.arange(len(a)))


class PassThroughDict(frozendict):
    def __missing__(self, key):
        return key

    def inverse_map(self, key):
        if key not in self.values():
            return key

        values = [k for k, v in self.items() if v == key]
        if len(values) == 0:
            raise IndexError('Key has no inverse.')
        elif len(values) > 1:
            raise IndexError('Key has too many inverses.')
        elif key not in self.keys():
            raise IndexError('Key has too many inverses (one of which is an implicit passthrough).')
        else:
            return values[0]


class StreamingEstimator(ABC):
    def __init__(self, input_streams=None, output_streams=None, log_level=None):
        """
        Parameters
        ----------
        input_streams: dict
            Keys are stream numbers, values are a flag to the transformer about how to process the data.
            So {3: 'X'} would mean that stream 3 should be processed as an X variable.
            Data not in an input_stream will usually be passed through.
        output_streams: dict[int, int]
            Keys are input streams, values are output streams; this is stream remapping applied after the transformer.
        log_level: int
            0: no logging
            1: profiling
            2: basic logging
            3: complete logging
        """

        self.input_streams = PassThroughDict(input_streams or {})
        self.output_streams = PassThroughDict(output_streams or {})
        self.log_level = log_level or 0
        self.mid_run_sources = None
        self.log = dict(step_time=[], stream=[])


    def step(self, data, stream=0, return_output_stream=False):
        """
        Learns and applies a transformation to incoming data.

        Parameters
        ----------
        data: any, np.ndarray
            data can be anything, but for most transformers it will be an array of shape (n_samples, sample_dimension)
        stream: int | typing.Hashable
            The stream the incoming data is coming from; 0 is the default.
            While this could technically be any hashable value, the convention is to use ints.
        return_output_stream: bool
            Whether to return the output stream; this is mostly only useful in pipelines, and so is false by default.

        Returns
        -------
        data
            the processed data
        stream: int, optional
            the stream the outputted data should be routed to
        """
        if self.log_level >= 1:
            start = time.time()
            self.log['stream'].append(stream)
            self.pre_log_for_step(data, stream)

        ret = self._step(data, stream, return_output_stream)

        if self.log_level >= 1:
            time_elapsed = time.time() - start
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

            self.log_for_step(data, stream)
        return ret

    def pre_log_for_step(self, data, stream):
        pass

    def log_for_step(self, data, stream):
        pass


    @abstractmethod
    def _step(self, data, stream, return_output_stream):
        # most implementations will need to handle initialization and nan values; possibly also logging?
        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def blank_copy(self):
        return type(self)(**self.get_params())

    def trace_route(self, stream):
        middle_str = str(self) if stream in self.input_streams else ""
        if stream == self.output_streams[stream]:
            return middle_str
        return [stream, middle_str, self.output_streams[stream]]

    def _parse_sources(self, sources):
        if not (isinstance(sources, tuple) or isinstance(sources, list)):  # passed a single source
            sources = [sources]
        elif not len(sources): # passed an empty list
            warnings.warn('passed an empty sources list')
            return [], []

        if not isinstance(sources[0], tuple):  # passed a list of sources without streams
            streams = range(len(sources))
            sources = zip(sources, streams)

        sources, streams = zip(*sources)


        new_sources = []
        for source in sources:
            if isinstance(source, np.ndarray) and not isinstance(source, ArrayWithTime):
                source = ArrayWithTime.from_notime(source)
            elif not isinstance(source, np.ndarray):
                source = GeneratorDataSource(source)

            if isinstance(source, ArrayWithTime):
                source = copy.deepcopy(source)
                if len(source.shape) == 2:
                    source = source[:,None,:]
                    assert source.shape[0] == len(source.t)

            new_sources.append(source)
        sources = new_sources

        return sources, streams


    def streaming_run_on(self, sources, return_output_stream=False):
        """
        Parameters
        ----------
        sources: np.ndarray, types.GeneratorType, list[np.ndarray | types.GeneratorType], DataSource, list[DataSource], list[tuple[DataSource, int]], dict
            This should be the set of data sources.
            Inputs are parsed like this:
                a single array gets upgraded to a list: a -> [a]
                a list gets zipped with `range()`:  [a] -> [(a,0)]
                the elements returned from iter(a) will get fed into the 0 stream
        return_output_stream: bool
            Whether to yield the output stream or not. This is false by default to not confuse first-time users.

        Yields
        -------
        data: np.ndarray
            The processed version of each element of the given iterator.
        stream: int, optional
            the stream that the outputted data belongs to
        """

        sources, streams = self._parse_sources(sources)

        sources = list(zip(map(iter, sources), streams))
        self.mid_run_sources = sources
        while True:  # while-true/break is a code smell, but I want a do-while
            next_time = float('inf')
            for source, stream in reversed(sources):  # reversed to prefer the first element
                source_next_time = source.next_sample_time()
                if source_next_time <= next_time:
                    next_time = source_next_time
                    next_source, next_stream = source, stream
            if not next_time < float('inf'):
                break

            yield self.step(data=next(next_source), stream=next_stream, return_output_stream=return_output_stream)

        self.mid_run_sources = None

    def offline_run_on(self, sources, convinient_return=True, exit_time=None, show_tqdm=False):
        outputs = {}

        exit_time_for_tqdm = float('inf') if exit_time is None else exit_time

        pre_pbar = contextlib.nullcontext()
        if show_tqdm:
            for source in self._parse_sources(copy.deepcopy(sources))[0]:
                if hasattr(source, 't'):
                    exit_time_for_tqdm = min(exit_time_for_tqdm, source.t.max())
            pre_pbar = tqdm(total=None if exit_time_for_tqdm == float('inf') else round(exit_time_for_tqdm,2))

        with pre_pbar as pbar:
            for data, stream in self.streaming_run_on(sources, return_output_stream=True):
                if exit_time is not None and data.t > exit_time:
                    break
                if stream not in outputs:
                    outputs[stream] = []
                outputs[stream].append(data)
                if show_tqdm:
                    assert not isinstance(data.t, np.ndarray) or data.t.size == 1
                    pbar.update(round(float(data.t), 2) - pbar.n)

        if convinient_return:
            if isinstance(convinient_return, bool):
                convinient_return = 0

            if convinient_return not in outputs:
                warnings.warn(f"No outputs were routed to stream '{convinient_return}'.")
                outputs[convinient_return] = []

            data = outputs[convinient_return]
            outputs = ArrayWithTime.from_list(data, squeeze_type='to_2d', drop_early_nans=True)  # can be replaced with np.squeeze

        return outputs


    def __str__(self):
        kwargs = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({kwargs})"

    # for printing and testing
    def get_params(self, deep=True):
        # TODO: should this deep copy?
        return dict(input_streams=self.input_streams, output_streams=self.output_streams, log_level=self.log_level)

    # this is mostly for testing
    def expected_data_streams(self, rng, DIM, cycles=1):
        for _ in range(cycles):
            for s in self.input_streams:
                yield rng.normal(size=(10, DIM)), s

    @property
    def base_algorithm(self):
        """
        This is mostly for testing; it's useful for checking that e.g. ProSVD (the transformer) has the same arguments
        as BaseProSVD (which is not a transformer.)
        """
        return type(self)


class DecoupledEstimator(StreamingEstimator):
    def __init__(self, *, input_streams=None, output_streams=None, log_level=None):
        super().__init__(input_streams, output_streams, log_level)
        self.frozen = False

    def _step(self, data, stream=0, return_output_stream=False):
        self.partial_fit(data, stream)
        return self.transform(data, stream, return_output_stream)

    def partial_fit(self, data, stream=0) -> None:
        if self.frozen:
            return
        self._partial_fit(data, stream)

    @abstractmethod
    def _partial_fit(self, data, stream):
        """data should be of shape (n_samples, sample_size)"""
        # TODO: implement common functionality here
        pass

    @abstractmethod
    def transform(self, data, stream=0, return_output_stream=False):
        pass

    def freeze(self, b=True):
        self.frozen = b

    def offline_fit_then_transform(self, sources, convinient_return=True, exit_time=None):
        self.offline_run_on(sources, convinient_return, exit_time)
        self.freeze()
        return self.offline_run_on(sources, convinient_return, exit_time)

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        raise NotImplementedError()



class Pipeline(DecoupledEstimator):
    def __init__(self, steps=(), *, input_streams=None, reroute_inputs=True, output_streams=None, log_level=None):
        self.steps: list[DecoupledEstimator] = steps
        self.reroute_inputs = reroute_inputs

        if input_streams is None:
            if reroute_inputs:
                expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
                input_streams = dict(zip(range(len(expected_streams)), expected_streams))
            else:
                input_streams = PassThroughDict({})

        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)

    def get_params(self, deep=True):
        p = dict(steps=self.steps, reroute_inputs=self.reroute_inputs)
        if deep:
            for i, step in enumerate(self.steps):
                for k, v in step.get_params(deep).items():
                    p[f'__steps[{i}]__{k}'] = v
        return p | super().get_params(deep)

    def _partial_fit(self, data, stream=0):
        self.step(data, stream)

    def _step(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.step(data, stream=stream, return_output_stream=True)

        stream = self.output_streams[stream]
        if not return_output_stream:
            return data
        return data, stream

    def transform(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.transform(data, stream=stream, return_output_stream=True)
        stream = self.output_streams[stream]

        if not return_output_stream:
            return data
        return data, stream

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        stream = self.output_streams.inverse_map(stream)
        for step in self.steps[::-1]:
            data, stream = step.inverse_transform(data, stream=stream, return_output_stream=True)
        stream = self.input_streams.inverse_map(stream)

        if not return_output_stream:
            return data

        return data, stream

    def freeze(self, b=True):
        self.frozen = b
        for step in self.steps:
            step.freeze(b)

    def trace_route(self, stream):
        super_path = [stream]

        path = []
        stream = self.input_streams[stream]
        for step in self.steps:
            path.append(step.trace_route(stream))
            stream = step.output_streams[stream]

        super_path.append(path)
        stream = self.output_streams[stream]
        super_path.append(stream)

        if super_path[0] == super_path[2]:
            return path
        return super_path

    def __str__(self):
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.steps)}])"


class Predictor(StreamingEstimator):
    stream_to_update_log_on = None
    def __init__(self, input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1):
        input_streams = input_streams or {0: 'X', 1: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.check_dt = check_dt
        self.dt = None
        self._last_X_t = None
        self.parameter_fitting = True

        self.n_steps_to_predict = n_steps_to_predict
        self.unevaluated_log_pred_ps = {}
        self.predictions = {}

    @abstractmethod
    def predict(self, n_steps):
        pass

    @abstractmethod
    def observe(self, X, stream=None):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_arbitrary_dynamics_parameter(self):
        """returns nan if unitialized"""
        pass

    @abstractmethod
    def unevaluated_log_pred_p(self, n_steps):
        pass


    def step(self, data, stream=0, return_output_stream=False):
        original_data = None
        if self.log_level >= 2:
            original_data = copy.deepcopy(data)

        if self.log_level >= 1:
            self.log['stream'].append(stream)

        start = time.time()
        ret = self._step(data, stream, return_output_stream)
        time_elapsed = time.time() - start

        if self.log_level >= 1:
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

        self.log_for_step(data, stream, original_data=original_data)
        return ret

    def log_for_step(self, data, stream, original_data=None):
        if self.log_level >= 2:
            assert self.check_dt
            if 'pred_error' not in self.log:
                for k in ['pred_error', 'log_pred_p', 'log_pred_p_origin_t', 'pred_origin_t']:
                    self.log[k] = []

            if self.dt is not None:
                if self.input_streams[stream] == 'X':
                    current_t = data.t
                    real_time_offset = self.dt * self.n_steps_to_predict

                    # normal error calculation
                    for t_to_eval in list(self.predictions.keys()):
                        if np.isclose(t_to_eval - current_t, 0, atol=self.dt/10):
                            origin_t, prediction = self.predictions[t_to_eval]
                            self.log['pred_error'].append(ArrayWithTime(prediction - original_data, current_t))
                            self.log['pred_origin_t'].append(origin_t)
                            del self.predictions[t_to_eval]
                        elif t_to_eval < current_t:
                            del self.predictions[t_to_eval]

                    # log pred p calculation
                    for t_to_eval in list(self.unevaluated_log_pred_ps.keys()):
                        if np.isclose(t_to_eval - current_t, 0, atol=self.dt/10):
                            origin_t, pdf = self.unevaluated_log_pred_ps[t_to_eval]
                            self.log['log_pred_p'].append(ArrayWithTime(pdf(original_data), current_t))
                            self.log['log_pred_p_origin_t'].append(origin_t)
                            del self.unevaluated_log_pred_ps[t_to_eval]
                        elif t_to_eval < current_t:
                            del self.unevaluated_log_pred_ps[t_to_eval]

                    self.predictions[current_t + real_time_offset] = (current_t, self.predict(self.n_steps_to_predict))
                    self.unevaluated_log_pred_ps[current_t + real_time_offset] = (current_t, self.unevaluated_log_pred_p(self.n_steps_to_predict))


    def toggle_parameter_fitting(self, value=None):
        if value is not None:
            self.parameter_fitting = bool(value)
        else:
            self.parameter_fitting = not self.parameter_fitting

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            if self.check_dt:
                assert hasattr(data, 't')
                if self._last_X_t is not None:
                    dt = data.t - self._last_X_t
                    assert dt > 0
                    if self.dt is not None:
                        consistent_dt = np.isclose(data.t - self._last_X_t, self.dt)
                        # assert consistent_dt, 'time steps for training are not consistent'
                        if not consistent_dt:
                            warnings.warn('time steps for training are not consistent')
                        self.dt = (self.dt + dt)/2
                    else:
                        self.dt = dt
                self._last_X_t = data.t

            data_depth = 1
            assert data.shape[0] == data_depth

            if np.isfinite(data).all():
                self.observe(data, stream=stream)
            else:
                warnings.warn('there should probably be an autonomous dynamics call here')

            data = ArrayWithTime.from_transformed_data(self.get_state().reshape(data_depth,-1), data)

        elif self.input_streams[stream] == 'dt_X':
            steps = self.data_to_n_steps(data)
            pred = self.predict(n_steps=steps)
            data = ArrayWithTime.from_transformed_data(pred, data)
        elif self.input_streams[stream] == 'toggle_parameter_fitting':
            self.toggle_parameter_fitting(data)

        return (data, stream) if return_output_stream else data

    def data_to_n_steps(self, data):
        assert data.size == 1
        q_dt = data[0, 0]
        if self.check_dt and self.dt is not None:
            steps = q_dt / self.dt
        else:
            steps = q_dt

        assert np.isclose(steps, steps := round(steps)), "without tracking dt, queries must be an integer number of steps"
        steps = int(steps)
        return steps

    def make_prediction_times(self, source, n_steps=1):
        dt = (source.dt if self.check_dt else 1) * n_steps
        return ArrayWithTime(np.ones_like(source.t).reshape(-1,1) * dt, source.t)

    @staticmethod
    def plot_pdf(fig, ax, pdf_f, xlim, ylim, native_d=3, e1=None, e2=None, density=100, add_colorbar=True):
        # TODO: move this to be a standalone in plotting_functions
        if e1 is None or e2 is None:
            assert e1 is None and e2 is None
            e1 = np.zeros(native_d)
            e2 = np.zeros(native_d)
            e1[0] = 1
            e2[1] = 1
        elif isinstance(e1,int):
            assert isinstance(e2,int)
            pre_e1 = np.zeros(native_d)
            pre_e2 = np.zeros(native_d)
            pre_e1[e1] = 1
            pre_e2[e2] = 1
            e1, e2 = pre_e1, pre_e2

        x_bins = np.linspace(*xlim, density + 1)
        y_bins = np.linspace(*ylim, density + 1)
        pdf_values = np.zeros(shape=(density, density))
        for i in range(density):
            for j in range(density):
                x = (x_bins[i] + x_bins[i + 1]) / 2
                y = (y_bins[j] + y_bins[j + 1]) / 2
                pdf_values[i, j] = pdf_f(x * e1 + y * e2)
        pdf_values = np.array(pdf_values)

        im = ax.pcolormesh(x_bins, y_bins, pdf_values.T, cmap='plasma')
        if add_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(check_dt=self.check_dt, n_steps_to_predict=self.n_steps_to_predict)


    def expected_data_streams(self, rng, DIM, cycles=1):
        dt = 1
        start_t = self._last_X_t or -1
        for i in range(1, cycles+1):
            yield ArrayWithTime(rng.normal(size=(1, DIM)), t=i*dt + start_t), 'X'
            yield ArrayWithTime(np.ones((1, 1)) * dt, t=i*dt+ start_t), 'dt_X'
            yield ArrayWithTime(np.ones((1, 1)) * (rng.random() > .9), t=i*dt+ start_t), 'toggle_parameter_fitting'


class TypicalEstimator(DecoupledEstimator):
    def __init__(self, *, input_streams=None, output_streams=None, log_level=None, on_nan_width=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.is_initialized = False
        self.on_nan_width = on_nan_width

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p = self.instance_get_params() | {'on_nan_width': self.on_nan_width} | p
        return p

    def _partial_fit(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if np.isnan(data).any():
                idx = np.isnan(data).any(axis=1)
                if idx.all():
                    return
                data = data[~np.isnan(data).any(axis=1)]

            if not self.is_initialized:
                self.pre_initialization_fit_for_X(data)
            else:
                self.partial_fit_for_X(data)

    def transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized or np.isnan(data).any():
                if self.on_nan_width is None:
                    data = np.nan * data
                else:
                    data = (np.nan * data)[:,:self.on_nan_width]
            else:
                data = self.transform_for_X(data)

        stream = self.output_streams[stream]
        if return_output_stream:
            return data, stream
        return data

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        stream = self.output_streams.inverse_map(stream)
        if self.input_streams[stream] == 'X':
            if not self.is_initialized or np.isnan(data).any():
                data = np.nan * data
            else:
                data = self.inverse_transform_for_X(data)

        if return_output_stream:
            return data, stream
        return data

    def pre_initialization_fit_for_X(self, X):
        self.is_initialized = True

    @abstractmethod
    def partial_fit_for_X(self, X):
        pass

    @abstractmethod
    def transform_for_X(self, X):
        pass

    @abstractmethod
    def instance_get_params(self, deep=True):
        pass

    def inverse_transform_for_X(self, X):
        raise NotImplementedError()


class CenteringEstimator(TypicalEstimator):
    def __init__(self, *, init_size=0, input_streams=None, output_streams=None, nan_when_uninitialized=False, on_nan_width=None, log_level=None):
        super().__init__(input_streams=input_streams, output_streams=output_streams, on_nan_width=on_nan_width, log_level=log_level)
        self.init_size = init_size
        self.samples_seen = 0
        self.center = 0
        self.nan_when_uninitialized = nan_when_uninitialized

    def pre_initialization_fit_for_X(self, X):
        self.partial_fit_for_X(X)
        if self.samples_seen >= self.init_size:
            self.is_initialized = True

    def partial_fit_for_X(self, X):
        self.samples_seen += X.shape[0]
        self.center = self.center + (X.sum(axis=0) - X.shape[0] * self.center) / self.samples_seen

    def transform_for_X(self, X):
        if not self.is_initialized and self.nan_when_uninitialized:
            return np.nan * X
        else:
            return X - self.center

    def inverse_transform_for_X(self, X):
        return X + self.center

    def instance_get_params(self, deep=True):
        return {'init_size': self.init_size, 'nan_when_uninitialized': self.nan_when_uninitialized}


class KernelSmoother(StreamingEstimator):
    def __init__(self, *, tau=1, kernel_length=None, custom_kernel=None, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0:'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.tau = tau
        self.kernel_length = kernel_length
        self.custom_kernel = custom_kernel
        if custom_kernel is None:
            delta_t = 1 # todo: make time-aware
            alpha = 1 - np.exp(-delta_t/tau)
            if kernel_length is None:
                kernel_length = np.ceil(tau * 5).astype(int)

            kernel = alpha * (1-alpha)**np.arange(kernel_length)[::-1]
        else:
            kernel = custom_kernel
        self.kernel = kernel
        self.last_X = None
        self.history = deque(maxlen=len(self.kernel))

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            output = []
            for row in data:
                self.history.append(row)
                if len(self.history) >= len(self.kernel) and not np.isnan(a:=np.array(self.history)).any():
                    output.append(self.kernel @ a)
                else:
                    output.append(np.nan*row)
            data = ArrayWithTime.from_transformed_data(output, data)
        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def get_params(self, deep=True):
        return dict(tau=self.tau, kernel_length=self.kernel_length, custom_kernel=self.custom_kernel) | super().get_params()



# proSVD.py

import scipy.linalg


def align_column_spaces(A, B):
    # https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, (B).T

def is_orthonormal(Q, rows_too=False):
    o = np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
    if rows_too:
        o = o and np.allclose(Q @ Q.t, np.eye(Q.shape[0]))
    return o

def principle_angles(Q1, Q2):
    assert is_orthonormal(Q1) and is_orthonormal(Q2)
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    return np.arccos(np.clip(s, -1, 1))

def column_space_distance(Q1, Q2, method='angles', override_ortho_check=False):
    if not override_ortho_check:
        for Q in Q1, Q2:
            assert is_orthonormal(Q)
    else:
        warnings.warn('this method is intended to be used for only orthogonal matrices')

    if method == 'angles':
        return np.abs(principle_angles(Q1, Q2)).sum()
    elif method == 'aligned_diff':
        Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
        return np.linalg.norm(Q1_rotated - Q2)
    else:
        raise ValueError()


class BaseProSVD:
    def __init__(self, k=None, decay_alpha=None, whiten=None):
        self.k = k or 1
        self.decay_alpha = decay_alpha or 1
        self.whiten = whiten or False

        self.Q = None
        self.R = None
        self.n_samples_observed = 0

    def initialize(self, x):
        sample_d, n_samples = x.shape

        assert n_samples >= self.k, "please init with # of cols >= k"
        assert sample_d >= self.k, "k size doesn't make sense"

        Q, R_diag, _ = np.linalg.svd(x, full_matrices=False)

        self.Q = Q[:, :self.k]
        self.R = np.diag(R_diag[:self.k])
        self.n_samples_observed = n_samples

    def add_new_input_channels(self, n):
        if self.Q is not None:
            self.Q = np.vstack([self.Q, np.zeros(shape=(n, self.Q.shape[1]))])

    def updateSVD(self, x):
        x_along = self.Q.T @ x
        x_orth = x - self.Q @ x_along
        x_orth_q, x_orth_r = np.linalg.qr(x_orth, mode='reduced')

        q_new = np.hstack([self.Q, x_orth_q])
        r_new = np.block([[self.R, x_along], [np.zeros((x_orth_r.shape[0], self.R.shape[1])), x_orth_r]])  # 2x2 block matrix

        try:
            u_high_d, diag_high_d, vh_high_d = np.linalg.svd(r_new, full_matrices=False)
        except np.linalg.LinAlgError:
            u_high_d, diag_high_d, vh_high_d = scipy.linalg.svd(r_new, full_matrices=False, lapack_driver='gesvd')


        u_low_d = u_high_d[:,:self.k]
        vh_low_d = vh_high_d[:,:self.k]
        diag_low_d = diag_high_d[:self.k]

        diag_low_d *= self.decay_alpha

        # if 'alignment_method' == 'procrustean':

        # The new basis is `q_new @ u_low_d`; to align it to `X` we would do the SVD of `X.T @ (q_new @ u_low_d)`.
        # Since we want to align to `self.Q`, we would usually use `self.Q.T @ q_new @ u_low_d`, but we can simplify
        # because (self.Q.T @ q_new) has a lot of cancellations (see their definitions).
        temp = np.linalg.svd(u_low_d[:self.k, :], full_matrices=False)
        u_stabilizing_rotation = temp[0] @ temp[2]
        u_low_d_stabilized = u_low_d @ u_stabilizing_rotation.T

        # TODO: we don't actually stabilize anything here, I think this can be dropped
        vh_low_d_stabilized, vh_stabilizing_rotation = scipy.linalg.rq(vh_low_d)

        # elif 'alignment_method' == 'Baker 2012':
        #     # Baker refers to e.g. https://doi.org/10.1016/j.laa.2011.07.018
        #     u_low_d_stabilized, u_stabilizing_rotation = rq(u_low_d)
        #     vh_low_d_stabilized, vh_stabilizing_rotation = rq(vh_low_d)
        # elif 'alignment_method' == 'sequential KLT':
        #     # KLT is in the original proSVD code, not sure what the source is
        #     u_low_d_stabilized = u_low_d
        #     u_stabilizing_rotation = u_low_d.T @ u_low_d  # identity matrix
        #
        #     vh_low_d_stabilized = vh_low_d
        #     vh_stabilizing_rotation = vh_low_d.T @ vh_low_d

        self.Q = q_new @ u_low_d_stabilized
        self.R = (u_stabilizing_rotation * diag_low_d) @ vh_stabilizing_rotation.T

        self.n_samples_observed *= self.decay_alpha
        self.n_samples_observed += x.shape[1]

    def project_down(self, x):
        ret = self.Q.T @ x
        if self.whiten:
            # todo: this can be sped up with lapack.dtrtri or linalg.solve
            R = self.R / np.sqrt(self.n_samples_observed)
            ret = np.linalg.inv(R) @ ret
        return ret

    def project_up(self, x):
        if self.whiten:
            R = self.R / np.sqrt(self.n_samples_observed)
            x = R @ x
        return self.Q @ x

    def get_cov_matrix(self, low_d=False):
        R = self.R / np.sqrt(self.n_samples_observed)
        if low_d:
            return R @ R.T
        else:
            return self.Q @ R @ R.T @ self.Q.T



class proSVD(TypicalEstimator, BaseProSVD):
    base_algorithm = BaseProSVD

    def __init__(self, *, init_size=None, k=None, decay_alpha=None, whiten=None, input_streams=None, output_streams=None, on_nan_width=None, log_level=None):
        TypicalEstimator.__init__(self, input_streams=input_streams, output_streams=output_streams, on_nan_width=on_nan_width, log_level=log_level)
        BaseProSVD.__init__(self, k=k, decay_alpha=decay_alpha, whiten=whiten)
        self.init_size = init_size or self.k * 2
        self.on_nan_width = self.k
        self.init_samples = []
        self.is_partially_initialized = False
        self.log |= {'Q': [], 't': []}


    def instance_get_params(self, deep=True):
        return dict(k=self.k, decay_alpha=self.decay_alpha, whiten=self.whiten, init_size=self.init_size)


    def pre_initialization_fit_for_X(self, X):
        if not self.is_partially_initialized:
            self.init_samples += list(X)
            if len(self.init_samples) >= self.init_size:
                width_max = np.squeeze([s.shape for s in self.init_samples]).max(axis=0)
                init_array = np.zeros((len(self.init_samples), width_max))
                for i, s in enumerate(self.init_samples):
                    init_array[i,:s.size] = s
                self.initialize(np.squeeze(init_array).T)
                self.is_partially_initialized = True
        else:
            self.updateSVD(X.T)
        if self.is_partially_initialized and (not self.whiten or np.linalg.matrix_rank(self.R) == self.R.shape[0]):
            self.is_initialized = True


    def transform_for_X(self, X):
        return self.project_down(X.T).T

    def inverse_transform_for_X(self, X):
        return self.project_up(X.T).T

    def partial_fit_for_X(self, X):
        if X.shape[1] > self.Q.shape[0]:
            self.add_new_input_channels(X.shape[1] - self.Q.shape[0])
        self.updateSVD(X.T)

    def log_for_step(self, data, stream=0):
        if self.is_initialized:
            if self.log_level >= 2:
                self.log['Q'].append(ArrayWithTime(self.Q, data.t))

    def get_distance_from_subspace_over_time(self, subspace):
        assert self.log_level >= 2
        m = len(self.log['Q'])
        distances = np.empty(m)
        for j, Q in enumerate(self.log['Q']):
            if np.any(np.isnan(Q)):
                distances[j] = np.nan
                continue
            distances[j] = ArrayWithTime(column_space_distance(Q, subspace, method='angles'), Q.t)
        distances = ArrayWithTime.from_list(distances)
        return distances

    def get_Q_stability(self):
        assert self.log_level >= 2
        Qs = ArrayWithTime.from_list(self.log['Q'])

        dQ = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
        dQ = ArrayWithTime(dQ, Qs.t[1:])
        return dQ

    def plot_Q_stability(self, ax):
        """
        Parameters
        ----------
        ax: matplotlib.axes.Axes
            the axes on which to plot the history
        """
        dQ = self.get_Q_stability()
        ax.plot(dQ.t, dQ)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\Vert dQ_i\Vert$')
        ax.set_title(f'Change in the columns of proSVD Q over time ({self.Q.shape[0]} -> {self.Q.shape[1]})')

    @classmethod
    def offline_run_on_and_cache(cls, input_arr, **kwargs):
        pro = cls(**kwargs)
        return pro.offline_run_on(input_arr, convinient_return=True)






import functools
from collections import deque

import numpy as np
from scipy.stats import multivariate_normal

class KalmanFilter:
    # TODO: make this a transformer once fit
    def __init__(self, use_steady_state_k=False, subtract_means=True):
        self.use_steady_state_K = use_steady_state_k
        self.subtract_means = subtract_means

        self.A = None  # state transitions
        self.C = None  # link between states and observations
        self.W = None  # state noise
        self.Q = None  # observation noise

        self.X_mean = None
        self.Y_mean = None

        self.steady_state_K = None

        self.state_var = None
        self.state = None

    def fit(self, X, Y):
        _X = None
        if isinstance(X, list):
            assert len(np.array(X[0]).shape) == 3
            _X = X
            X, Y = np.vstack([np.vstack(x) for x in X]), np.vstack([np.vstack(y) for y in Y])

        X_mean, Y_mean = (X.mean(axis=0), Y.mean(axis=0)) if self.subtract_means else (0,0)

        X = X - X_mean
        Y = Y - Y_mean

        origin = X[:-1]
        destination = X[1:]
        if _X is not None:
            if max([len(x) for x in _X]) == 1:
                warnings.warn("not fitting because there isn't enough data")
                return
            origin = np.vstack([np.vstack(x[:-1]) for x in _X])
            destination = np.vstack([np.vstack(x[1:]) for x in _X])
        A, _, _, _ = np.linalg.lstsq(origin, destination)

        C, _, _, _ = np.linalg.lstsq(X, Y)

        w = X[1:] - X[:-1] @ A
        W = (w.T @ w) / (X.shape[1] - 1)

        q = Y - X @ C
        Q = (q.T @ q) / (X.shape[1])

        # model variables
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        self.X_mean = X_mean
        self.Y_mean = Y_mean

        if self.use_steady_state_K:
            m = X.shape[1]
            P = W
            matrix = C.T @ P @ C + Q

            K_old = P @ C @ np.linalg.pinv(matrix)
            P = (np.eye(m) - C @ K_old.T) @ P
            for i in range(3000):
                P = A @ P @ A.T + W
                matrix = C.T @ P @ C + Q
                K = P @ C @ np.linalg.pinv(matrix)
                P = (np.eye(m) - C @ K.T) @ P

                dif = np.abs(K - K_old)
                K_old = K
                if (dif < 1E-16).all():
                    break
            self.steady_state_K = K

        # state variables
        self.state = np.zeros_like(X[-1:])
        self.state_var = self.W

    @staticmethod
    def inference_step(state, state_var, *, A, C, W, Q, X_mean, Y_mean, Y=None, kalman_gain=None):
        state = state - X_mean
        state = state @ A
        state_var = A @ state_var @ A.T + W

        if Y is not None:
            Y = Y - Y_mean
            if kalman_gain is None:
                kalman_gain = state_var @ C @ np.linalg.pinv(C.T @ state_var @ C + Q)
            state = state + (Y - state @ C) @ kalman_gain.T
            state_var = (np.eye(C.shape[0]) - C @ kalman_gain.T) @ state_var

        return state + X_mean, state_var


    def kf_step(self, Y=None):
        self.state, self.state_var = self.inference_step(self.state, self.state_var, Y=Y, A=self.A, C=self.C, W=self.W, Q=self.Q, Y_mean=self.Y_mean, X_mean=self.X_mean, kalman_gain=None if not self.use_steady_state_K else self.steady_state_K)
        return self.state

    def predict_state_and_var(self, n_steps, state, state_var):
        prediction = np.zeros((n_steps+1, self.A.shape[0])) * np.nan
        prediction_var = np.zeros((n_steps+1, self.A.shape[0], self.A.shape[0])) * np.nan
        prediction[0] = state
        prediction_var[0] = state_var
        for i in range(n_steps):
            state, state_var = self.inference_step(state, state_var, Y=None, A=self.A, C=self.C, W=self.W, Q=self.Q, Y_mean=self.Y_mean, X_mean=self.X_mean, kalman_gain=None if not self.use_steady_state_K else self.steady_state_K)
            prediction[i+1] = state
            prediction_var[i+1] = state_var

        return prediction, prediction_var

    def predict(self, n_steps, initial_state=None, initial_state_var=None):
        state = initial_state if initial_state is not None else self.state
        state_var = initial_state_var if initial_state_var is not None else self.state_var
        prediction, prediction_var = self.predict_state_and_var(n_steps, state, state_var)
        return prediction


class StreamingKalmanFilter(Predictor, KalmanFilter):
    base_algorithm = KalmanFilter
    def __init__(self, *, steps_between_refits = 25, use_steady_state_k=False, subtract_means=True, no_hidden_state=True, input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1, max_history_length=5000):
        input_streams = input_streams or {0: 'X', 1: 'Y', 2: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        Predictor.__init__(self, input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)
        KalmanFilter.__init__(self, use_steady_state_k=use_steady_state_k, subtract_means=subtract_means)
        self.no_hidden_state = no_hidden_state
        self.steps_between_refits = steps_between_refits
        self.max_history_length = max_history_length

        self.last_seen = {}
        self.latent_state_history = [[]]
        self.observation_history = [[]]

    def predict(self, n_steps):
        if self.A is not None:
            predicted_latent_state = KalmanFilter.predict(self, n_steps)[-1]
            predicted_observation = (predicted_latent_state @ self.C)
        else:
            predicted_observation = np.array([[np.nan]])
        return predicted_observation

    def observe(self, X, stream=None):
        semantic_stream = self.input_streams[stream]
        if semantic_stream in {'X', 'Y'}:
            if self.parameter_fitting:
                self.last_seen[semantic_stream] = X

            if ('Y' in self.last_seen or self.no_hidden_state) and 'X' in self.last_seen and self.parameter_fitting:
                self.observation_history[-1].append(self.last_seen['X'])
                self.latent_state_history[-1].append(self.last_seen['X' if self.no_hidden_state else 'Y'])

            if semantic_stream == 'X' and self.A is not None:
                self.kf_step(X)

            assert len(self.latent_state_history[-1]) == len(self.observation_history[-1])
            n_seen = sum(len(x) if len(x) > 1 else 0 for x in self.observation_history)
            if (
                    n_seen % self.steps_between_refits == 0
                    and len(self.observation_history[-1]) > 1
                    and self.parameter_fitting
            ):
                self.fit(X=self.latent_state_history, Y=self.observation_history)
                latent = np.squeeze(self.latent_state_history[-1])
                obs = np.squeeze(self.observation_history[-1])

                while sum([len(x) for x in self.observation_history]) > self.max_history_length:
                    if len(self.observation_history[0]) == 2:
                        self.observation_history.pop(0)
                        self.latent_state_history.pop(0)
                    else:
                        self.observation_history[0].pop(0)
                        self.latent_state_history[0].pop(0)


                constant = min(self.steps_between_refits, len(obs)) # TODO: set this more rigorously
                self.state = latent[obs.shape[0]-constant]
                for i in range(constant):
                    self.kf_step(Y=obs[obs.shape[0] - constant + i])

    def toggle_parameter_fitting(self, value=None):
        before = self.parameter_fitting
        super().toggle_parameter_fitting(value)
        if before and not self.parameter_fitting:
            self.last_seen = {}
            if len(self.latent_state_history[-1]) > 1:
                self.latent_state_history.append([])
            else:
                self.latent_state_history[-1] = []

            if len(self.observation_history[-1]) > 1:
                self.observation_history.append([])
            else:
                self.observation_history[-1] = []

    def get_state(self):
        state = self.state if self.state is not None else np.array([np.nan])
        return state

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(use_steady_state_k=self.use_steady_state_K, subtract_means=self.subtract_means, steps_between_refits=self.steps_between_refits)

    def get_arbitrary_dynamics_parameter(self):
        if self.A is None:
            return np.nan

        return self.A


    def unevaluated_log_pred_p(self, n_steps):
        if self.A is None:
            return lambda x: np.nan

        evals, evecs  = np.linalg.eigh(self.state_var)
        evals = np.abs(evals)  # TODO: this should not be necessary
        state_var = (evecs * evals) @ evecs.T

        state = np.array(self.state)
        A = np.array(self.A)
        C = np.array(self.C)
        W = np.array(self.W)
        Q = np.array(self.Q)
        X_mean = np.array(self.X_mean)
        Y_mean = np.array(self.Y_mean)

        inner_state = state
        inner_state_var = state_var
        for i in range(n_steps):
            inner_state, inner_state_var = KalmanFilter.inference_step(inner_state, inner_state_var, A=A, C=C, W=W, Q=Q, X_mean=X_mean, Y_mean=Y_mean)
        try:
            rv = multivariate_normal(mean=inner_state.flatten(), cov=inner_state_var)
        except np.linalg.LinAlgError:
            warnings.warn("covariance matrix is not positive definite")
            return lambda x: np.nan

        return rv.logpdf

class MultiKernelRegressor:
    def __init__(self, length_scales=(1e-1,1e-1,1e-9), maxlen=100, input_names=('stim_location', 'stim_vector', 'stim_time'), reweight_every=1, rng=None):
        self.maxlen = maxlen
        self.input_histories = None
        self.output_history = None
        self.n_observed = 0
        self.input_names = input_names
        self.reweight_every = reweight_every
        if rng is None:
            rng = numpy.random.default_rng(0)
        self.rng = rng
        self.log = {'length_scales': [], 'preq_errors':[]}

        self.length_scales = numpy.array(length_scales)

    def observe(self, x, y):
        if any([numpy.any(~numpy.isfinite(sub_x)) for sub_x in x]) or numpy.any(~numpy.isfinite(y)):
            warnings.warn("ignoring non-finite input")
            return

        self.log['preq_errors'].append(y - self.predict(x))

        if self.input_histories is None:
            self.input_histories = [numpy.zeros(shape=(self.maxlen, sub_x.size)) * numpy.nan for sub_x in x]
            self.output_history = numpy.zeros(shape=(self.maxlen, y.size))

        if self.n_observed == self.maxlen:
            warnings.warn("history is full, overwriting old observations")
        index = self.n_observed % self.maxlen

        for history, sub_x in zip(self.input_histories, x):
            history[index, :] = sub_x
        self.output_history[index, :] = y
        self.n_observed += 1

        if self.n_observed % self.reweight_every == 0:
            self.reweight()

    def reweight(self):
        if self.n_observed < 2:
            return
        sample_size = min(self.n_observed, 15)
        sample = self.rng.permutation(min(self.n_observed, self.maxlen))[:sample_size]
        external_weight_vec = numpy.ones(self.maxlen)
        external_weight_vec[sample] = 0
        f = self.make_jax_pred_f()
        def evaluate(length_scales):
            if numpy.any(length_scales <= 1e-10) or numpy.any(length_scales > 1e6):
                return numpy.inf

            errors = numpy.zeros(sample_size)
            for i, idx in enumerate(sample):
                try:
                    errors[i] = numpy.linalg.norm(f([h[idx] for h in self.input_histories], length_scales, weight_modifiers=external_weight_vec) - self.output_history[idx])**2
                except (OverflowError, ZeroDivisionError):
                    errors[i] = numpy.inf
            return numpy.mean(errors)

        current = evaluate(self.length_scales)
        new_length_scales = numpy.array(self.length_scales)

        coefs = numpy.logspace(-1, 1, 5)
        for i in range(len(self.length_scales)):
            errors = numpy.zeros(5)
            for j, coef in enumerate(coefs):
                if coef == 1:
                    errors[j] = current
                    continue
                test_length_scales = numpy.array(self.length_scales)
                test_length_scales[i] *= coef
                errors[j] = evaluate(test_length_scales)
            new_length_scales[i] *= coefs[numpy.argmin(errors)]

        self.log['length_scales'].append(numpy.array(self.length_scales))
        lr = 0.05
        self.length_scales = numpy.exp(numpy.log(self.length_scales) * lr + numpy.log(new_length_scales) * (1-lr))

    def plot_length_scales(self, ax):
        for series, label in zip(numpy.array(self.log['length_scales']).T, self.input_names):
            ax.plot(series, label=label + ' curvy')
        ax.semilogy()


    def make_jax_pred_f(self):
        # TODO: precompute
        if self.input_histories is None:
            def f(x):
                return numpy.array([[numpy.nan]])
        else:
            input_histories = [jnp.nan_to_num(h, nan=jnp.inf) for h in self.input_histories]
            output_history = jnp.array(self.output_history)
            ones = jnp.ones(len(self.output_history))
            def f(x, length_scales=jnp.array(self.length_scales), weight_modifiers=ones):
                log_weights = 0
                for (sub_x, history, length_scale) in zip(x, input_histories, length_scales):
                    distances = jnp.linalg.norm(history - jnp.squeeze(sub_x), axis=1)
                    distances = jnp.nan_to_num(distances, nan = jnp.inf)
                    log_weights += -length_scale * jnp.square(distances)
                log_weights = jnp.nan_to_num(log_weights, nan=-numpy.inf, neginf=-numpy.inf)
                log_sum = jax.scipy.special.logsumexp(log_weights, b=weight_modifiers)
                log_weights = log_weights - log_sum

                return jnp.exp(jnp.clip(log_weights, max=0, min=-30)) @ output_history
        return f

    def predict(self, x):
        return numpy.array(self.make_jax_pred_f()(x))

    def get_obs(self, i=None, t=None):
        """gets last by default"""
        if t is not None: # use time
            assert i is None
            candidates = numpy.nonzero(numpy.abs(t - self.input_histories[self.input_names.index('stim_time')].flatten()) < 1e-12)
            assert len(candidates) == 1
            assert len(candidates[0]) == 1
            i = candidates[0][0]
        else: # use i
            if i is None: # get last obs
                i = (self.n_observed - 1) % self.maxlen
        return {k:v[i] for k, v in zip(self.input_names, self.input_histories)} | {'output': self.output_history[i]}

# TODO: make the time comparisons more uniform

dt_epsilon = 1e-8

class StimAutoReg():
    def __init__(self, n_steps_to_consider):
        self.n_steps_to_consider = n_steps_to_consider
        self.previous_corrections = []
        self.training_data = []
        self.coeffs = np.zeros(n_steps_to_consider) * np.nan

    def correct(self, current_t, dt):
        new_correction = 0
        for correction in reversed(self.previous_corrections):
            steps = (current_t - correction.t) / dt
            # assert abs(steps - round(steps)) < dt/4, steps # TODO: is this ok to ignore?
            steps = int(round(steps))

            if steps >= self.n_steps_to_consider:
                break
            new_correction += correction * self.coeffs[steps-1]
        return new_correction

    def observe_new_correction(self, new_correction):
        self.previous_corrections.append(np.squeeze(new_correction))
        self.training_data.append([])

    def observe(self, X, pred_callback, dt):
        if len(self.previous_corrections) == 0:
            return

        steps = (X.t - self.previous_corrections[-1].t)/dt
        if steps >= self.n_steps_to_consider + 2: # TODO: simplify this logic
            return
        steps = int(round(steps))
        if not abs(steps - round(steps)) < dt/5:  # TODO: make this standard
            print(f'{steps=} {X.t=}')
            raise Exception()
        if steps >= self.n_steps_to_consider + 1:
            return

        pred = pred_callback()
        residual = X - pred
        self.training_data[-1].append(np.squeeze(residual))

        if len(self.training_data) > 1 and type(self.training_data[-2]) is list:
            if len(self.training_data[-2]) == self.n_steps_to_consider:
                self.training_data[-2] = self.training_data[-2]
            else:
                self.training_data.pop(-2)
            errors = np.array(self.training_data[:-1])
            corrections = np.array(self.previous_corrections[:-1])[:,None,:]

            corrections = corrections.transpose((0,2,1))
            errors = errors.transpose((0,2,1))
            self.coeffs, _, _, _ = np.linalg.lstsq(corrections.reshape((-1, 1)), errors.reshape((-1, self.n_steps_to_consider)))
            self.coeffs = self.coeffs.flatten()

class StimRegressor(Predictor):
    stream_to_update_log_on = 'stim'
    def __init__(self, autoreg=None, stim_reg=None, heed_stimuli=True, attempt_correction=True, error_on_missed_stim=True, input_streams=None, output_streams=None, log_level=None, check_dt=True, n_steps_to_predict=1, stim_delay=0):
        input_streams = input_streams or {0: 'X', 1: 'stim', 2: 'dt_X'}
        assert n_steps_to_predict == 1
        assert heed_stimuli or not attempt_correction  # correcting without learning doesn't make sense
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)

        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        if stim_reg is None:
            stim_reg = MultiKernelRegressor(maxlen=100)
        self.stim_reg: MultiKernelRegressor = stim_reg
        self.attempt_correction = attempt_correction
        self.heed_stimuli = heed_stimuli
        self.last_seen_stims = deque()
        self.stim_autoreg = StimAutoReg(n_steps_to_consider=0)
        assert stim_delay >= 0
        self.stim_delay = stim_delay  # in units of time (wrt the data)
        self.error_on_missed_stim = error_on_missed_stim

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            if self.is_notable_stim(data):
                self.last_seen_stims.append(data)
            ret =  (data, stream) if return_output_stream else data
        else:
            ret = super()._step(data, stream, return_output_stream)

        if hasattr(data, 't'):
            self.trim_last_seen_stims(current_t=data.t)

        return ret

    @staticmethod
    def is_notable_stim(stim):
        return (stim!=0).any()

    def in_stim_lag(self, current_t):
        for stim in self.last_seen_stims:
            dt = self.dt
            if dt is None:
                dt = self.stim_delay # TODO: is this a good idea?
            if stim.t + self.stim_delay + dt/10 >= current_t and self.is_notable_stim(stim):
                return True
        return False

    def trim_last_seen_stims(self, current_t):
        saftey_margin = self.dt*1.2 if self.dt else self.stim_delay # TODO: check this timing/synchronization logic
        while self.last_seen_stims and (current_t - self.last_seen_stims[0].t) > (self.stim_delay + saftey_margin):
            if self.error_on_missed_stim and self.heed_stimuli and np.isfinite(self.autoreg.get_arbitrary_dynamics_parameter()).all():
                raise MissedStimulusError(f"Missed stim. {current_t=:.3f} {self.last_seen_stims[0].t=:.3f} (diff={current_t-self.last_seen_stims[0].t:.2f}) {(self.stim_delay + saftey_margin)=:.3f}")
            self.last_seen_stims.popleft()

    def get_stim_to_correct_for(self, current_t, remove=False):
        to_return = []
        for stim in self.last_seen_stims:
            if np.isclose(stim.t + self.stim_delay, current_t, atol=self.dt/20):
                to_return.append(stim)

        if remove:
            for stim in to_return:
                self.last_seen_stims.remove(stim)

        if len(to_return) == 0:
            return []
        elif len(to_return) == 1:
            return to_return[0].flatten()
        else:
            raise Exception("Can only correct for one stimulus at a time.")



    def log_for_step(self, data, stream, original_data=None):
        super().log_for_step(data, stream, original_data=original_data)

        if self.log_level >= 2 and self.dt is not None:
            if self.input_streams[stream] == 'stim':
                real_time_offset = self.dt * self.n_steps_to_predict
                assert self.n_steps_to_predict == 1
                current_t_as_of_last_x = self._last_X_t
                prediction_time = current_t_as_of_last_x + real_time_offset
                for saved_prediction_time in self.predictions.keys():
                    if np.isclose(current_t_as_of_last_x - saved_prediction_time, current_t_as_of_last_x - prediction_time, rtol=.05):
                        prediction_time = saved_prediction_time

                self.predictions[prediction_time] = (current_t_as_of_last_x, self.predict(self.n_steps_to_predict))
                self.unevaluated_log_pred_ps[prediction_time] = (current_t_as_of_last_x, self.unevaluated_log_pred_p(self.n_steps_to_predict))


    def predict_stim_response(self, stim_to_correct_for, current_t):
        # TODO: is current_t correct here?
        stim_reg_input = [self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for, current_t]
        return self.stim_reg.predict(stim_reg_input)

    def observe(self, X, stream=None):
        if self.heed_stimuli and self.in_stim_lag(current_t=X.t):
            self.autoreg.toggle_parameter_fitting(False)

            stim_to_correct_for = self.get_stim_to_correct_for(current_t=X.t, remove=True)
            if len(stim_to_correct_for):
                self.autoreg.toggle_parameter_fitting(False)
                pred = self.autoreg.predict(n_steps=1)
                residual = X - pred
                stim_reg_input = [self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for, X.t]  # TODO: deal with nan from autoreg
                self.stim_reg.observe(stim_reg_input, residual)
                self.stim_autoreg.observe_new_correction(ArrayWithTime(self.stim_reg.predict(stim_reg_input), X.t))

            # TODO: make a decision about wheither autoreg needs to be a transformer
            # self.autoreg.observe(X, stream=self.input_streams[stream])
            self.autoreg.step(data=X, stream=self.input_streams[stream])
        else:
            self.autoreg.toggle_parameter_fitting(True)
            self.stim_autoreg.observe(X,functools.partial(self.autoreg.predict,n_steps=1), self.dt)
            self.autoreg.step(data=X, stream=self.input_streams[stream])

    def get_state(self):
        return self.autoreg.get_state()

    def get_arbitrary_dynamics_parameter(self):
        return self.autoreg.get_arbitrary_dynamics_parameter()

    def predict(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        assert n_steps in {0,1}
        pred = self.autoreg.predict(n_steps=n_steps)

        if self.attempt_correction and np.isfinite(pred).all():
            current_t = current_t + self.dt * n_steps
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if len(stim_to_correct_for):
                pred = pred + self.predict_stim_response(stim_to_correct_for, current_t)
            pred = pred + self.stim_autoreg.correct(current_t, self.dt)
        return pred

    def unevaluated_log_pred_p(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        assert n_steps in {0,1}
        f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)

        if self.attempt_correction:
            current_t = self.dt * n_steps + current_t
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if len(stim_to_correct_for):
                correction = self.predict_stim_response(stim_to_correct_for, current_t)
            else:
                correction = 0
            def corrected_f(future_point):
                return f(future_point - correction)
        else:
            corrected_f = f
        return corrected_f

    def finalize_log(self, stim_intended_samples=None):
        self.log['pred_error'] = ArrayWithTime.from_list(self.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')
        if stim_intended_samples is not None:
            self.log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg, attempt_correction=self.attempt_correction, heed_stimuli=self.heed_stimuli, stim_delay=self.stim_delay, error_on_missed_stim=self.error_on_missed_stim)

    def __getstate__(self):
        # TODO: check for jax?
        self.unevaluated_log_pred_ps = {}
        return super().__getstate__()


class MissedStimulusError(RuntimeError):
    pass

del np

import time
import numpy
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize, LBFGS
import itertools
import copy
import warnings
from enum import Enum

class OptimizationMethod(str, Enum):
    JAXOPT = 'jaxopt'
    PREV_SEEN = 'prev_seen'
    CHEAT_LOWD_VEC = 'cheat_lowd_vec'
    CHEAT_HIGHD_VEC_SINGLE_NEURONS = 'cheat_highd_vec_single_neurons'
    CHEAT_HIGHD_VEC_MANY_NEURONS = 'cheat_highd_vec_many_neurons'  # TODO: this isn't really cheating, change the name?


class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=0,  # TODO: make this an rng
            should_log=False,
            lam_1=0.001,
            inter_stim_interval_generator=None,
            optimization_method=OptimizationMethod.JAXOPT,
            stim_timing_method='regular',
            initial_nostim_period=1,
            u_to_s_model_type='identity', # TODO: remove? it's used in sim_stim_design_stim
            n_random_initialization=1,
    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        assert max_l0_norm > 0
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1
        self.u_to_s_model_type = u_to_s_model_type

        self.optimization_method: OptimizationMethod = optimization_method
        self.n_random_initialization = n_random_initialization
        self.stim_timing_method = stim_timing_method
        self.initial_nostim_period = initial_nostim_period

        if inter_stim_interval_generator is None:
            inter_stim_interval_generator = itertools.repeat(1)
        self.inter_stim_interval_generator = inter_stim_interval_generator
        self.last_stim_time = None
        self.current_isi = None

        self.log = []

        self.objective_history = []

    def stim_when_extreme(self, current_t, objective_value):
        self.objective_history.append(objective_value)
        return current_t > 50 and objective_value == numpy.nanmin(self.objective_history)

    def decide_whether_to_stim(self, current_t, **kwargs):
        if current_t < self.initial_nostim_period:
            return False

        if self.stim_timing_method == 'isi':  # or 'regular'
            if self.last_stim_time is None:
                self.last_stim_time = self.initial_nostim_period if self.initial_nostim_period is not None else 0
                self.current_isi = next(self.inter_stim_interval_generator)
            if current_t > self.last_stim_time + self.current_isi:
                self.last_stim_time = current_t
                self.current_isi = next(self.inter_stim_interval_generator)
                return True
            return False
        elif self.stim_timing_method == 'extreme':
            return self.stim_when_extreme(current_t, **kwargs)
        elif self.stim_timing_method == 'random':
            return kwargs['stim_time_rng'].random() < 1/next(self.inter_stim_interval_generator) * kwargs['input_array_dt']
        else:
            raise ValueError()


    def desired_stim_direction(self, equivalent_projection_matrix, stim_direction_type, rng):  # TODO: use built-in rng
        if stim_direction_type == 'first':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 1))
            desired_stim[0] = 1
        elif stim_direction_type == 'first2':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 2))
            desired_stim[0] = 1
            desired_stim[1] = 1
        elif stim_direction_type == 'col':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 1))
            desired_stim[rng.choice(equivalent_projection_matrix.shape[1]), 0] = 1
        elif stim_direction_type == 'random':
            desired_stim = rng.normal(size=(equivalent_projection_matrix.shape[1], 1))
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        elif stim_direction_type == 'random+':
            desired_stim_high_d = rng.normal(size=(equivalent_projection_matrix.shape[0], 1))
            desired_stim_high_d = desired_stim_high_d / numpy.linalg.norm(desired_stim_high_d)
            desired_stim_high_d = numpy.abs(desired_stim_high_d)
            desired_stim = equivalent_projection_matrix.T @ desired_stim_high_d
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        elif stim_direction_type == 'random_feasible':
            desired_stim_high_d = rng.normal(size=(equivalent_projection_matrix.shape[0], 1))
            desired_stim_high_d = desired_stim_high_d / numpy.linalg.norm(desired_stim_high_d)
            desired_stim_high_d = numpy.abs(desired_stim_high_d).flatten()
            while (desired_stim_high_d > 0).sum() > self.max_l0_norm:
                desired_stim_high_d[rng.choice(len(desired_stim_high_d))] = 0
            desired_stim = equivalent_projection_matrix.T @ desired_stim_high_d
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
            desired_stim = desired_stim.reshape([-1,1])
        elif stim_direction_type == 'ones':
            desired_stim_high_d = numpy.ones((equivalent_projection_matrix.shape[0], 1))
            desired_stim = equivalent_projection_matrix.T @ desired_stim_high_d
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        elif stim_direction_type == '-ones':
            desired_stim_high_d = -numpy.ones((equivalent_projection_matrix.shape[0], 1))
            desired_stim = equivalent_projection_matrix.T @ desired_stim_high_d
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        else:
            raise ValueError()
        return desired_stim

    def register_stim(self):
        pass

    def design_stim_prev_seen(self, v, previous_us, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = lambda u: u

        # TODO: keep this consistent with jaxopt version
        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = 0
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        best_u = None
        best_loss = float('inf')
        # TODO: parallellize this
        for u in previous_us:
            loss = objective(u)
            if loss < best_loss:
                best_loss = loss
                best_u = u

        best_u = best_u / best_u.max()
        return best_u, {'s': u_to_s_function(u)}

    def design_stim_jaxopt(self, v, u_dimension, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = lambda x: x

        u = self.rng.uniform(size=(u_dimension,)) * .1

        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        lb = jnp.zeros_like(u)
        ub = jnp.ones_like(u)

        bounds = (lb, ub)
        intermediate_xs = []
        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
        result = runner.run(u, bounds=bounds)
        u = numpy.array(result.params)

        if u.max() > 0:
            u = numpy.array(u / u.max())


        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0

        return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}

    # design_stim_jaxopt_unconstrained
    def design_stim_jaxopt_unconstrained(self, v, u_dimension, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = lambda x: x

        u = self.rng.uniform(size=(u_dimension,)) * .1

        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = 0
            # loss += self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        # lb = jnp.zeros_like(u)
        # ub = jnp.ones_like(u)
        #
        # bounds = (lb, ub)
        intermediate_xs = []
        # runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', callback=lambda xk: intermediate_xs.append(xk) if self.should_log else None)
        # result = runner.run(u, bounds=bounds)

        runner = LBFGS(fun=objective)
        result = runner.run(u)
        u = numpy.array(result.params)

        if numpy.abs(u).max() > 0:
            u = numpy.array(u / u.max())


        # idx = numpy.argsort(u)
        # u[idx[:-self.max_l0_norm]] = 0

        return u, {'s': u_to_s_function(u), 'intermediate_xs': numpy.array(intermediate_xs)}



    def design_stim(self, v, optimization_method=None, **kwargs):
        start_time = time.time()
        assert len(v.shape) == 2

        l = {}
        if optimization_method is None:
            optimization_method = self.optimization_method

        match optimization_method:
            case OptimizationMethod.JAXOPT:
                u, l = self.design_stim_jaxopt(v, kwargs['u_dimension'], kwargs['u_to_s_function'])
            case OptimizationMethod.PREV_SEEN:
                u, l = self.design_stim_prev_seen(v, kwargs['previous_us'], kwargs['u_to_s_function'])
            case OptimizationMethod.CHEAT_LOWD_VEC:
                u = (kwargs['equivalent_projection_matrix'] @ v).flatten()
            case OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0])] = 1
            case OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS:
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0], size=self.max_l0_norm, replace=False)] = 1
            case _:
                raise ValueError()


        if self.should_log:
            self.log.append({
                                'optimization_time': time.time() - start_time,
                                'v':v,
                                'u':u,
                                's': numpy.nan * v
                            } | l)

        return u

    def sim_stim_design_stim(self, sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t):
        self: StimDesigner
        optimization_method = self.optimization_method
        u_to_s_model_type = self.u_to_s_model_type
        if sr.stim_reg.n_observed <= self.n_random_initialization and (u_to_s_model_type == 'kernel_regressed' or optimization_method == 'prev_seen'):
            # u_to_s_model_type = 'identity'
            u_to_s_model_type = None
            optimization_method = 'cheat_highd_vec_many_neurons'


        if optimization_method in {'jaxopt', 'prev_seen'}:
            stim_reg = sr.stim_reg
            previous_us = stim_reg.input_histories[1][:stim_reg.n_observed] if optimization_method == 'prev_seen' else None
            if u_to_s_model_type == 'kernel_regressed':
                f = stim_reg.make_jax_pred_f()
                pred = sr.autoreg.predict(n_steps=0)
                def u_to_s_function(u):
                    return stim_magnitude * f([pred, u, current_t])
                designed_stim = self.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
            elif u_to_s_model_type == 'identity':
                def u_to_s_function(u):
                    return stim_magnitude * equivalent_projection_matrix.T @ u
                designed_stim = self.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0], previous_us=previous_us)
        elif optimization_method == 'cheat_lowd_vec' and u_to_s_model_type == 'identity':
            designed_stim = self.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix)
        elif optimization_method in {'cheat_highd_vec_single_neurons','cheat_highd_vec_many_neurons'} and u_to_s_model_type is None:
            designed_stim = self.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix, optimization_method=optimization_method)
        else:
            raise ValueError()

        self.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)
        self.log[-1]['time_of_stim'] = current_t
        self.log[-1]['equiv_proj_mat'] = equivalent_projection_matrix

        if (designed_stim == 0).all():
            designed_stim[0] = 1e-10
            warnings.warn("Stimulus was all zero!")  # TODO: handle this better

        return designed_stim










from improv.actor import Actor
from queue import Empty
import numpy as np
import time
import logging
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

    def setup(self):
        pass

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
        data = self.pro.partial_fit(data)

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
