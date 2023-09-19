from pathlib import Path
import json
from typing import Optional
import optuna
from optuna.trial import TrialState
from torchhep.optuna.objective import ObjectiveBase
from torchhep.optuna.plotting import plot_study


def summairze_study(study: optuna.Study,
                    output_path: Path | str | None,
                    verbose: bool = True
):
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[TrialState.COMPLETE])

    result = {
        "best": {
            "value": study.best_value,
            "params": study.best_trial.params,
            "number": study.best_trial.number,
        },
        "statistics": {
            "finished": len(study.trials),
            "pruned": len(pruned_trials),
            "complete": len(complete_trials),
        }
    }

    if output_path is not None:
        with open(output_path, 'w') as stream:
            json.dump(result, stream, indent=4)

    if verbose:
        print(json.dumps(result, indent=4))


def run_study(objective: ObjectiveBase,
              log_dir: Path,
              n_trials: int = 100,
              timeout: Optional[int] = None,
              name: str = 'study',
) -> None:
    storage_path = log_dir / 'study.db'
    storage = f'sqlite:///{storage_path}'
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(storage=storage, sampler=sampler,
                                pruner=pruner, study_name=name,
                                direction=objective.direction)

    study.optimize(func=objective, n_trials=n_trials, timeout=timeout)
    summairze_study(study, log_dir / 'summary.json')

    plot_dir = log_dir / 'plot'
    plot_dir.mkdir()
    plot_study(study, metric_name=objective.metric_name, output_dir=plot_dir)
