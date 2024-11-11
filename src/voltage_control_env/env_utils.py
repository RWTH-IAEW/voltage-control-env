from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import copy

import gymnasium as gym
import numpy as np
from shapely import Polygon, LineString, Point

from .dataset import DataSet, SimbenchDataSet

class ScenarioManager:
    '''
    Class wrapping the information about the scenarios provided to the environment.
    '''
    def __init__(self,
                 net, # pandapower net
                 dataset: DataSet,
                 ctrl_sgen_idx: List[int] = None,
                 pq_areas: List[ConstrainedPQArea] | ConstrainedPQArea | str = 'cone',
                 pv_converter_oversize: int = 1.1):
        '''
        Initializes the Scenario Manager
        Parameters:
            pp_net: Underlying pandapower network.
            dataset: Dataset holding the scenario profiles.
            control_sgen_idx: List of indices of the controllable (= observable) sgens. If None all the sgens are controllable.
            pq_areas: List of feasible (constrained) pq-areas for every controllable sgen (same length as control_sgen_idx).
                      If single pq-area or pq-area type as string, the given pq-area is applied to every controllable sgen.
            pv_converter_oversize: pv_converter_oversize: The oversizing factor of the PV converter in relation to the maximum p: S_max = P_max * pv_converter_oversize
        '''
        self.net = net
        self.dataset = dataset
        if ctrl_sgen_idx:
            self.ctrl_sgen_indices = ctrl_sgen_idx
        else:
            # make all the sgens controllable
            self.ctrl_sgen_indices = list(range(len(net.sgen)))
        
        if isinstance(pq_areas, List):
            self.pq_areas = pq_areas
        elif isinstance(pq_areas, PQArea):
            self.pq_areas = [copy.deepcopy(pq_areas) for _ in self.ctrl_sgen_indices]
        elif isinstance(pq_areas, str):
            self.pq_areas = [create_PQ_area(pq_areas) for _ in self.ctrl_sgen_indices]

        self.pv_converter_oversize = pv_converter_oversize

        self.no_agents = len(self.ctrl_sgen_indices)

        self._init_pv_converter()

    def _init_pv_converter(self,):
        '''
        Sets the converter capacity for all sgens according to formula: S_max = pv_converter_oversize * P_max
        '''

        # get the total maximum p over the entire dataset
        # TODO: This is not a very efficient solution...
        p_total_max = -1 * np.ones(shape=(len(self.net.sgen),))
        for idx in range(self.dataset.get_length()):
            max_p_mw = self.dataset.get_item(idx)['sgen']['max_p_mw']
            p_total_max = np.maximum(p_total_max, max_p_mw)
        
        # compute the s_max
        s_max = p_total_max * self.pv_converter_oversize

        # write data to network
        self.net.sgen['sn_mva'] = s_max


class ObservationGenerator(ABC):
    '''
    Base class responsible for generating observation for every controllable sgen.
    '''
    def __init__(self, net_ref, control_sgen_idx) -> None:
        '''
        Initializes the ObservationGenerator. 
        Parameters:
            net_ref: The underlying pandapower network
            control_sgen_idx: The indices of the controlled (= observed) sgens
        '''
        self.net = net_ref
        self.ctrl_sgen_indices = control_sgen_idx

    @abstractmethod
    def generate_observation(self,):
        pass
    
    @property
    @abstractmethod
    def observation_space(self,):
        '''The observation space for one single agent'''
        pass

class RewardGenerator(ABC):
    '''
    Base class responsible for generating a reward for every controllable sgen.
    '''
    def __init__(self, net_ref, control_sgen_idx) -> None:
        '''
        Initializes the RewardGenerator. 
        Parameters:
            net_ref: The underlying pandapower network
            control_sgen_idx: The indices of the controlled (= observed) sgens
        '''
        self.net = net_ref
        self.ctrl_sgen_indices = control_sgen_idx

    @abstractmethod
    def generate_reward(self,):
        pass


class StandardObservationGenerator(ObservationGenerator):
    '''
    Observation Manager considering the voltage, current active power injection, current reactive power injection
    and the maximum possible power injection per observable bus.
    '''
    def __init__(self, net_ref, control_sgen_idx) -> None:
        '''
        Initializes the ObservationGenerator. 
        Parameters:
            net_ref: The underlying pandapower network
            control_sgen_idx: The indices of the controlled (= observed) sgens
        '''
        super().__init__(net_ref, control_sgen_idx)

    def generate_observation(self,):
        '''
        Returns a nested dictionary of observations.
        Note: 
        The shape of the dictionary is as follows:
        sgen_0_id:
            v: vm_pu
            p_inj: p_mw
            q_inj: q_mvar
            max_p: max_p_mw
        sgen_1_id:
            ...
        ...
        '''
        obs = {}

        # generate a dictionary with the the observations: voltage, cur_p, cur_q, max_p for every sgen
        for sgen_idx in self.ctrl_sgen_indices:
            local_obs = {}
            bus = self.net.sgen.loc[sgen_idx, 'bus']

            local_obs['v'] = self.net.res_bus.loc[bus, 'vm_pu']
            local_obs['p_inj'] = self.net.sgen.loc[sgen_idx, 'p_mw']
            local_obs['q_inj'] = self.net.sgen.loc[sgen_idx, 'q_mvar']
            local_obs['max_p'] = self.net.sgen.loc[sgen_idx, 'max_p_mw']

            obs[sgen_idx] = local_obs

        return obs
    
    @property
    def observation_space(self,):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.ctrl_sgen_indices),))
    

class StandardRewardGenerator(RewardGenerator):
    '''
    Reward Generator penalizing the violation of the voltage bounds, rewards minimal curtailment of sgens
    '''
    def __init__(self, net_ref, control_sgen_idx, max_volt=1.05, min_volt=0.95, k_penalty = 100, k_power = 1) -> None:
        '''
        Initializes the Reward Generator.
        Parameters:
            net_ref: The underlying pandapower network
            control_sgen_idx: The indices of the controlled (= observed) sgens
            max_volt: the upper voltage bound (default=1.05pu)
            min_volt: the lower voltage bound (default=0.95pu)
            k_penalty: multiplicative factor penalizing deviation from voltage bounds
            k_power: mutliplicatve factor rewarding power usage
        '''
        super().__init__(net_ref, control_sgen_idx)
        self.max_volt = max_volt
        self.min_volt = min_volt
        self.k_penalty = k_penalty
        self.k_power = k_power

    def generate_reward(self):
        '''
        Returns a dictionary of rewards with one key-value pair for each controllable sgen.
        Note: 
        The shape of the dictionary is as follows:
        sgen_0_id: Reward_0
        sgen_1_id: Reward_1
        ...
        '''
        reward = {}

        # evaluate whether there are any voltage violations
        band_violations = np.abs(self.net.res_bus['vm_pu'] - np.clip(self.net.res_bus['vm_pu'], self.min_volt, self.max_volt))
        max_violation = np.max(band_violations)

        # In this case only give the penalty as reward
        if max_violation > 0:
            for ctrl_idx in self.ctrl_sgen_indices:
                reward[ctrl_idx] = self.k_penalty * -max_violation
        else:
            # compute the curtailment reward
            # TODO: Currently there is a discontinuity for the reward at 0. p = 0.1 / 0.1 is 1 but p = 0 / 0 is 0 
            norm_fac = self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] - self.net.sgen.loc[self.ctrl_sgen_indices, 'min_p_mw']
            rel_powers = (self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'] - self.net.sgen.loc[self.ctrl_sgen_indices, 'min_p_mw']) / (norm_fac + 1e-7)

            for ctrl_idx, rel_pow in zip(self.ctrl_sgen_indices, rel_powers):
                reward[ctrl_idx] = self.k_power * rel_pow

        return reward


class PQArea(ABC):
    """
    Base class for modeling feasible PQ areas
    """
    @abstractmethod
    def q_flexibility(self, p_pu, vm_pu=None):
        pass
    
    @abstractmethod
    def total_p_flexibility(self,):
        pass

    @abstractmethod
    def is_inside(self, p, q):
        pass
    

# until the next pandapower release copy this code here
class PQAreaPOLYGON(PQArea):
    """ Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points_pu' and 'q_points_pu'.

    Note: Due to generator point of view, negative q values are correspond with underexcited behavior.

    Example
    -------
    >>> PQAreaDefault(p_points_pu=(0.1, 0.2, 1, 1, 0.2, 0.1, 0.1),
    ...               q_points_pu=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1))
    """
    def __init__(self, p_points_pu, q_points_pu):
        self.p_points_pu = p_points_pu
        self.q_points_pu = q_points_pu

        self.polygon = Polygon([(p_pu, q_pu) for p_pu, q_pu in zip(p_points_pu, q_points_pu)])

    def in_area(self, p_pu, q_pu, vm_pu=None):
        return np.array([self.polygon.contains(Point(pi, qi)) for pi, qi in zip(p_pu, q_pu)])

    def q_flexibility(self, p_pu, vm_pu=None):
        def _q_flex(p_pu):
            line = LineString([(p_pu, -1), (p_pu, 1)])
            if line.intersects(self.polygon):
                points = [point[1] for point in LineString(
                    [(p_pu, -1), (p_pu, 1)]).intersection(self.polygon).coords]
                if len(points) == 1:
                    return [points[0]]*2
                elif len(points) == 2:
                    return points
                else:
                    raise ValueError(f"{len(points)=} is wrong. 2 or 1 is expected.")
            else:
                return [0, 0]
        return np.r_[[_q_flex(pi) for pi in p_pu]]
    
    def total_p_flexibility(self):
        """
        Computes the largest and lowest value for the self.polygon in the first dimension.
        Returns:
            np.ndarray: array of shape (2,) where the first element is the lowest value and the second element is the largest value.
        """
        # Extracting all points from the polygon
        points = self.polygon.exterior.coords
        # Separating the first dimension (p) from the points
        p_values = [point[0] for point in points]
        # Finding the minimum and maximum p values
        min_p = min(p_values)
        max_p = max(p_values)
        return np.r_[[min_p, max_p]]

    def is_inside(self, p, q):
        return self.in_area([p],[q])[0]
    
class ConstrainedPQArea(PQArea):
    '''
    PQ-area wrapper that constrains the area such that it does not allow regions outside the unit circle (adhering to maximum apparent power)
    '''
    def __init__(self, pq_area: PQArea):
        self.pq_area = pq_area

    def is_inside(self, p, q):
        in_pq_area = self.pq_area.is_inside(p, q)
        in_circle = np.sqrt(p**2 + q**2) <= 1

        return (in_pq_area and in_circle)
    
    def total_p_flexibility(self):
        return np.clip(self.pq_area.total_p_flexibility(), -1, 1)

    def q_flexibility(self, p_pu, vm_pu=None):
        q_flex = self.pq_area.q_flexibility(p_pu, vm_pu)
        q_flex_circle = np.vstack([-np.sqrt(1 - p_pu**2), np.sqrt(1-p_pu**2)])
        clipped_q_flex = np.clip(q_flex, q_flex_circle[0, :].reshape(-1, 1), q_flex_circle[1, :].reshape(-1, 1))

        return clipped_q_flex


def create_PQ_area(area_type: str):
    '''
    Factory function to create different (typical) PQ-areas.
    Currently supported areas: box, cone
    '''
    if area_type.lower() == 'box':
        return ConstrainedPQArea(PQAreaPOLYGON(p_points_pu=(1, 1, 0, 0), q_points_pu=(-1, 1, 1, -1)))
    if area_type.lower() == 'cone':
        return ConstrainedPQArea(PQAreaPOLYGON(p_points_pu=(1, 0, 1), q_points_pu=(-1, 0, 1)))
