from clustertools import Computation
from clustertools.storage import PickleStorage


class CliComputation(Computation):
    def __init__(self, exp_name, comp_name, main_fn, context="n/a",
                 storage_factory=PickleStorage, **env_params):
        """

        :param exp_name:
        :param comp_name:
        :param main_fn: main function taking sys.argv[1:] as only parameter
        :param context:
        :param storage_factory:
        :param env_params: environment parameters
        """
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._env_params = env_params
        self._main_fn = main_fn

    def run(self, results, **kwargs):
        args = []
        for job_param, value in kwargs.items():
            args.append("--" + job_param)
            args.append(str(value))
        for env_key, env_val in self._env_params.items():
            args.append("--" + env_key)
            args.append(str(env_val))
        for key, val in self._main_fn(args).items():
            results[key] = val


class CliComputationFactory(object):
    def __init__(self, fn, **kwargs):
        self._kwargs = kwargs
        self._fn = fn

    def __call__(self, exp_name, comp_name, context="n/a", storage_factory=PickleStorage, **kwargs):
        return CliComputation(exp_name, comp_name, self._fn, context=context, storage_factory=storage_factory, **self._kwargs)