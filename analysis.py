'''
                    Copyright Oliver Kowalke 2020.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import abc
import argparse
import ast
import configparser
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from functools import reduce
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.metrics import r2_score

class Metric(abc.ABC):
    @abc.abstractmethod
    def _process(seld, zone, denisty, din_z5):
        'process data'

    def __init__(self, zone, density, din_z5):
        self._process(zone, density, din_z5)

    @property
    def min_density(self):
        return self._min_density

    @property
    def max_density(self):
        return self._max_density

    @property
    def r_square(self):
        return self._r2

    @property
    def zone_at_min_density(self):
        return self._z_min_density

    @property
    def effective_din(self):
        return self._effective_din

    def poly(self, x):
        return self._p(x)

    @property
    def lin_min_zone_gamma(self):
        return self._z_min_gamma

    @property
    def lin_max_zone_gamma(self):
        return self._z_max_gamma

    @property
    def lin_gamma(self):
        return self._lin_gamma

    def linear(self, x):
        return self._lin_gamma * math.log10(2) * x + self._lin_intercept

    @property
    def n_gamma(self):
        return self._n_gamma

    def normal(self, x):
        return self._n_gamma * math.log10(2) * x + self._n_intercept

'metric based on: Weidner, Workshop'
class WskMetric(Metric):
    _min_density = 0.1
    _max_density = 1.3
    _z_min_gamma = 2
    _z_max_gamma = 8
    _n_gamma = 0.59
    _n_intercept = -0.124

    def __init__(self, zone, density, din_z5):
        super(WskMetric, self).__init__(zone, density, din_z5)

    def _process(self, zone, density, din_z5):
        self._zone = zone
        self._density = density
        # fitting polynom of 3rd grade
        self._p = np.poly1d(np.polyfit(zone, density, 3))
        # r^2 of polynomal fit
        self._r2 = round(r2_score(self._density, self._p(self._zone)), 4)
        # zone at min shadow density
        r = (self._p - self._min_density).r
        z = [x for x in r if x >= min(self._zone) and x <= max(self._zone)]
        self._z_min_density = round(z[0], 2) if 1 == len(z) else None
        # shift to eff. DIN
        self._effective_din = int(round(din_z5 - round((self._z_min_density - 1.0) * 3, 0), 0))
        # linear curve for points between zone _z_min_gamma and _z_max_gamma
        self._lin_gamma = (self._p(self._z_max_gamma)-self._p(self._z_min_gamma))/((self._z_max_gamma-self._z_min_gamma) * math.log10(2))
        self._lin_intercept = self._p(self._z_min_gamma) - self._lin_gamma * self._z_min_gamma * math.log10(2)


'metric based on: Lambrecht/Woodhouse, Way Beyond Monochrome'
class WbmMetric(Metric):
    _min_density = 0.17
    _max_density = 1.37
    _z_min_gamma = 1.5
    _z_max_gamma = 8.5
    _n_gamma = 0.57
    _n_intercept = -0.087

    def __init__(self, zone, density, din_z5):
        super(WbmMetric, self).__init__(zone, density, din_z5)

    def _process(self, zone, density, din_z5):
        self._zone = zone
        self._density = density
        # fitting polynom of 3rd grade
        self._p = np.poly1d(np.polyfit(zone, density, 3))
        # r^2 of polynomal fit
        self._r2 = round(r2_score(self._density, self._p(self._zone)), 4)
        # zone at min shadow density
        r = (self._p - self._min_density).r
        z = [x for x in r if x >= min(self._zone) and x <= max(self._zone)]
        self._z_min_density = round(z[0], 2) if 1 == len(z) else None
        # shift to eff. DIN
        self._effective_din = int(round(din_z5 - round((self._z_min_density - 1.5) * 3, 0), 0))
        # linear curve for points between zone _z_min_gamma and _z_max_gamma
        self._lin_gamma = (self._p(self._z_max_gamma)-self._p(self._z_min_gamma))/((self._z_max_gamma-self._z_min_gamma) * math.log10(2))
        self._lin_intercept = self._p(self._z_min_gamma) - self._lin_gamma * self._z_min_gamma * math.log10(2)


def normalize_zone(data, din_z5):
    din, density = zip(*data)
    return [round((din_z5 - float(x)) / 3 + 5, 1) for x in din], [float(x) for x in density]


def normalize_data(data, bf):
    data = [(x, ast.literal_eval(y)) for (x, y) in data]
    return[(x, round(round(reduce(lambda a, b: float(a) + float(b), y) / len(y), 3) - bf, 3)) for x, y in data]


def parse_data(file_p):
    config = configparser.ConfigParser()
    config.read(str(file_p))
    exposure = config['EXPOSURE']
    development = config['DEVELOPMENT']
    bf = 0
    if config.has_option('DATA', 'b+f'):
        bf = ast.literal_eval(config['DATA']['b+f'])
        bf = round(reduce(lambda a, b: float(a) + float(b), bf) / len(bf), 3)
        config.remove_option('DATA', 'b+f')
    data = config.items('DATA')
    data = normalize_data(data, bf)
    return exposure, development, data, bf


def plot(file_p, metric, zone, density, exposure, development, bf, show, wbm):
    fig, ax = plt.subplots()
    ax.set_title('{}: {}@{} {}[{}] {}/{} {}'.format(
        'Lambrecht/Woodhouse' if wbm else 'Weidner',
        exposure['film'],
        exposure['DIN'],
        development['developer'],
        development['dilution'],
        development['time'],
        development['temperature'],
        development['aggitation']),
        fontsize=10)
    ax.grid(which='both')
    ax.scatter(zone, density, marker='+', color='blue')
    ax.scatter(metric.zone_at_min_density, metric.min_density, marker='o', color='red', label='Zone(Dmin)={}'.format(metric.zone_at_min_density))
    line = np.linspace(zone[0], zone[-1], 200)
    ax.plot(line, metric.poly(line), linewidth=1.0, label='Polynom 3. Grades', color='blue')
    ax.plot(line, metric.linear(line), dashes=[1,1], label='linear (Zone {} und {})'.format(metric.lin_min_zone_gamma, metric.lin_max_zone_gamma))
    ax.plot(line, metric.normal(line), linewidth=1.0, label='ideal', color='green')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('Zone')
    ax.set_ylabel('Dichte über Schleier')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    handles, labels = ax.get_legend_handles_labels()
    if 0 < bf:
        desc = 'b+f = {}\neff. DIN = {}\nDmin({}) = {}\nDmax({}) = {}\nGamma = {}\nR^2= {}'.format(
            bf,
            metric.effective_din,
            metric.lin_min_zone_gamma,
            round(metric.poly(metric.lin_min_zone_gamma), 2),
            metric.lin_max_zone_gamma,
            round(metric.poly(metric.lin_max_zone_gamma), 2),
            round(metric.lin_gamma, 2),
            metric.r_square)
    else:
        desc = 'eff. DIN = {}\nGamma = {}\nR^2= {}'.format(
            metric.effective_din,
            round(metric.lin_gamma, 2),
            metric.r_square)
    handles.append(mpatches.Patch(color='none', label=desc))
    plt.legend(handles=handles, fontsize=8)
    if (show):
        plt.show()
    else:
        file_p = file_p.with_suffix('.png')
        file_p = file_p.parent.joinpath('{}-{}'.format(
            'fa' if wbm else 'std',
            file_p.name))
        plt.savefig(str(file_p), dpi=300)


def main(file_p, show, wbm):
    exposure, development, data, bf = parse_data(file_p)
    din_z5 = int(exposure['DIN'])
    zone, density = normalize_zone(data, din_z5)
    metric = WbmMetric(zone, density, din_z5) if wbm else WskMetric(zone, density, din_z5)
    plot(file_p, metric, zone, density, exposure, development, bf, show, wbm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.set_defaults(show=False)
    parser.add_argument('--wbm', dest='wbm', action='store_true')
    parser.set_defaults(wbm=False)
    args = parser.parse_args()
    file_p = Path(args.file).resolve()
    assert file_p.exists()
    main(file_p, args.show, args.wbm)
