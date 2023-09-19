from typing import Optional
from pathlib import Path
from itertools import combinations
import math
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as ovm


def plot_pretty_contour(study: optuna.Study,
                        param_0: str,
                        param_1: str,
                        target_name: Optional[str] = None,
) -> plt.Subplot:
    ax: plt.Subplot = ovm.plot_contour(
        study, params=[param_0, param_1])

    best_param_0 = study.best_params[param_0]
    best_param_1 = study.best_params[param_1]

    ax.scatter(best_param_0, best_param_1, s=500, marker='*', color='tab:red')

    default_xlow = ax.get_xlim()[0]
    default_ylow = ax.get_ylim()[0]

    guideline_kwargs = {'color': 'pink', 'ls': ':', 'lw': 2}
    ax.plot(2 * [best_param_0], [default_ylow, best_param_1],
            **guideline_kwargs)
    ax.plot([default_xlow, best_param_0], 2 * [best_param_1],
            **guideline_kwargs)

    ax.set_title('') # no title
    ax.set_xlabel(param_0, size=20)
    ax.set_ylabel(param_1, size=20)
    ax.tick_params(axis='both', labelsize=15)

    fig = ax.get_figure() # type: ignore
    if len(fig.get_axes()) > 1:
        colorbar_ax = fig.get_axes()[1] # type: ignore
        if target_name is not None:
            colorbar_ax.set_ylabel(target_name, size=20)
    return ax


def save_optuna_plot(ax: plt.Subplot,
                     output_path: Path,
                     figwidth: float = 12.8,
                     figheight: float = 9.6
):
    """
    Args:
        ax:
        output_path:
        figwidth: the optuna's default is 6.4
        figheight: the optuna's default is 4.8
    """
    fig: plt.Figure = ax.get_figure()
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)
    for suffix in ['.png', '.pdf']:
        fig.savefig(output_path.with_suffix(suffix))
    # FIXME
    plt.close(fig)


def plot_study(study: optuna.Study, target_name: str, output_dir: Path):
    """
    this function assumes that the study is a multi-objective optimization
    does not call 'plot_parallel_coordinate' and 'plot_slice' because contours
    are enough to understand what's happend.
    """
    def plot_and_save(name, **kwargs):
        func = getattr(ovm, 'plot_' + name)
        ax = func(study, **kwargs)
        save_optuna_plot(ax, output_dir / name)

    plot_and_save('edf', target_name=target_name)
    plot_and_save('optimization_history', target_name=target_name)
    plot_and_save('param_importances', target_name=target_name)
    plot_and_save('intermediate_values')

    # contour
    contour_dir = output_dir / 'contour'
    contour_dir.mkdir()

    param_dict = optuna.importance.get_param_importances(study)
    param_pair_list = list(combinations(param_dict.keys(), 2))
    pad_len = math.ceil(math.log10(len(param_pair_list)))
    output_name_template = f'{{idx:0>{pad_len}d}}__{{param_0}}__{{param_1}}'

    for idx, (param_0, param_1) in enumerate(param_pair_list):
        if param_dict[param_0] < param_dict[param_1]:
            param_0, param_1 = param_1, param_0
        ax = plot_pretty_contour(study, param_0, param_1)

        param_0 = param_0.replace('.', '-')
        param_1 = param_1.replace('.', '-')
        output_name = output_name_template.format(idx=idx,
                                                  param_0=param_0,
                                                  param_1=param_1)
        save_optuna_plot(ax, contour_dir / output_name)
