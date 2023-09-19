import dataclasses
from typing import Any
import optuna
from optuna.trial import Trial
from hierconfig.config import ConfigBase, config_field


def categorical_field(default, choices, help: str | None = None):
    if default not in choices:
        raise ValueError
    metadata = {
        'suggest': Trial.suggest_categorical,
        'kwargs': {
            'choices': choices,
        }
    }
    return config_field(default=default, metadata=metadata, help=help,
                        choices=choices)


def boolean_field(default, help: str | None = None):
    return categorical_field(default, [True, False], help)


def discrete_uniform_field(default: float, low: float, high: float, q: float,
                           help: str | None = None):
    """
    Args:
        q: A step of discretization
    """
    # TODO assert
    metadata = {
        'suggest': Trial.suggest_discrete_uniform,
        'kwargs': {
            'low': low,
            'high': high,
            'q': q
        }
    }
    return config_field(default=default, metadata=metadata, help=help)


def float_field(default, low, high, *, step=None, log=False,
                help: str | None = None):
    # TODO assert
    metadata = {
        'suggest': Trial.suggest_float,
        'kwargs': {
            'low': low,
            'high': high,
            'step': step,
            'log': log,
        }
    }
    return config_field(default=default, metadata=metadata, help=help)


def int_field(default, low, high, step=1, log=False, help: str | None = None):
    if not low <= default <= high:
        raise ValueError

    metadata = {
        'suggest': Trial.suggest_int,
        'kwargs': {
            'low': low,
            'high': high,
            'step': step,
            'log': log
        }
    }
    return config_field(default=default, metadata=metadata, help=help)


def loguniform_field(default, low, high, help: str | None = None):
    if not low <= default < high:
        raise ValueError
    metadata = {
        'suggest': Trial.suggest_loguniform,
        'kwargs': {
            'low': low,
            'high': high,
        }
    }
    return config_field(default=default, metadata=metadata, help=help)


@dataclasses.dataclass
class HyperparameterConfigBase(ConfigBase):

    @classmethod
    def _from_trial(cls, trial: optuna.Trial, prefixes: list[str]):
        kwargs: dict[str, Any] = {}

        for each in dataclasses.fields(cls):
            if issubclass(each.type, HyperparameterConfigBase):
                new_prefixes = prefixes.copy() + [each.name]
                kwargs[each.name] = each.type._from_trial(trial, new_prefixes)
            elif 'suggest' in each.metadata:
                name = '.'.join(prefixes + [each.name])
                kwargs[each.name] = each.metadata['suggest'](
                    trial, name, **each.metadata['kwargs'])
            elif each.default is not dataclasses.MISSING:
                kwargs[each.name] = each.default
            elif each.default_factory is not dataclasses.MISSING:
                kwargs[each.name] = each.default_factory()
            else:
                raise RuntimeError
        return cls(**kwargs)

    @classmethod
    def from_trial(cls, trial: optuna.Trial):
        return cls._from_trial(trial, [])


def hyperparameter(cls):
    """decorator"""
    fields = [(key, value) + ((getattr(cls, key), ) if hasattr(cls, key) else ())
              for key, value in cls.__annotations__.items()]
    return dataclasses.make_dataclass(cls.__name__, fields=fields,
                                      bases=(HyperparameterConfigBase, cls))
