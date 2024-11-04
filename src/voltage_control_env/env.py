from typing import List

import gymnasium as gym
import numpy as np
import pandapower as pp

from .dataset import DataSet, SimbenchDataSet
from .env_utils import ObservationGenerator, RewardGenerator, PQArea

class VoltageControlEnv(gym.Env):
    '''
    The Base Class of the Voltage Control Env implementing the Gymnasium Env interface.
    The environment is formulated in a more general multi-agent style and deviates from the standard Gymnasium specification.
    E.g. it returns a dictionary of rewards (one for every agent) instead of a single scalar value.

    In this environment the agent specifies his action by directly setting a desired setpoint in the feasible PQ-area.
    '''
    def __init__(self,
                 pp_net,
                 dataset: DataSet,
                 control_sgen_idx: List[int],
                 pq_areas: List[PQArea],
                 reward_generator : RewardGenerator,
                 observation_generator : ObservationGenerator,
                 random_setpoint_reset: bool = True):
        '''
        Initializes the environment.
        Parameters:
            pp_net: Underlying pandapower network.
            dataset: Dataset holding the possible scenarios.
            control_sgen_idx: List of indices of the controllable (= observable) sgens.
            pq_areas: List of feasible pq-areas for every controllable sgen (same length as control_sgen_idx)
            reward_generator: Reward Generator used for generating rewards.
            observation_generator: Observation Generator used for generating envrionment observations.
            random_setpoint_reset: Whether the setpoint is randomized (among feasible pq setpoints) upon reset or not.
        '''
        self.net = pp_net
        self.dataset = dataset
        self.ctrl_sgen_indices = control_sgen_idx
        self.pq_areas = pq_areas
        self.no_agents = len(control_sgen_idx)
        self.reward_generator = reward_generator
        self.observation_generator = observation_generator
        self.random_setpoint_reset = random_setpoint_reset
        #self.rng = np.random.default_rng(seed)
        self.scenario_id = None
        self.scenario = None

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.no_agents, 2))
        self.observation_space = observation_generator.observation_space  # per agent

    def reset(self, seed=None, options=None):
        '''
        Resets the environment.
        Parameters:
            seed: reseeds the environment (leave None for no reseeding)
            options: additional options regarding the environment reset. Supported options include:
                scenario_id: environment is reset following the provided id from the dataset.
                scenario: environment is reset following the provided scenario dict (e.g. for reproducibility of random setpoints upon reset)
                random_setpoint_reset: overwrites the random_setpoint_reset property of the environment.

        '''
        super().reset(seed=seed)
        scenario = None
        random_setpoint_reset = None

        if options is not None:
            if "scenario" in options:
                scenario = options['scenario']
            elif "scenario_id" in options:
                scenario = self.dataset.get_item(options['scenario_id'])

            if "random_setpoint_reset" in options:
                random_setpoint_reset = options['random_setpoint_reset']
            else:
                random_setpoint_reset = self.random_setpoint_reset

        # sample scenario if none was given
        if scenario is None:
            scenario = self._sample_scenario()
        
        # default to the environment random_setpoint_reset if none was given
        if random_setpoint_reset is None:
            random_setpoint_reset = self.random_setpoint_reset

        self._apply_scenario(scenario=scenario, random_setpoint_reset=random_setpoint_reset)
         
        # compute the resulting powerflow
        pp.runpp(self.net)

        # compute and return observation
        obs = self.observation_generator.generate_observation()

        return obs, {}

    def step(self, action):
        '''
        Progresses the environment by one step. Computing the successor observation and emitted reward.
        Parameters:
            action: numpy array of shape (num_control_sgens, 2) or (num_control_sgens * 2,)
                    relative real and reactive power injection per controlled sgen
        Returns:
            next_obs, reward, terminated, truncated, info
        '''
        action = self._handle_action(action)

        # set desired active and reactive power injections rel * max + (1-rel) * min
        rel_p_action = (action[:, 0] + 1) / 2
        p_flexibilities = np.vstack([pq_area.total_p_flexibility() for pq_area in self.pq_areas])
        effective_min_p = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] * p_flexibilities[:, 0], dtype=np.float64)
        effective_max_p = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] * p_flexibilities[:, 1], dtype=np.float64)

        p_inj = rel_p_action * effective_max_p + (1 - rel_p_action) * effective_min_p

        p_pu = rel_p_action * p_flexibilities[:, 1] + (1 - rel_p_action) * p_flexibilities[:, 0]
        q_flexibilities = np.vstack([pq_area.q_flexibility([p_pu[idx]]) for idx, pq_area in enumerate(self.pq_areas)])
        effective_min_q = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_q_mvar'] * q_flexibilities[:, 0], dtype=np.float64)
        effective_max_q = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_q_mvar'] * q_flexibilities[:, 1], dtype=np.float64)

        q_inj = (action[:, 1] + 1) / 2 * effective_max_q + (1 - (action[:, 1] +  1) / 2) * effective_min_q

        # apply to net
        self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'] = p_inj
        self.net.sgen.loc[self.ctrl_sgen_indices, 'q_mvar'] = q_inj

        # run power flow
        pp.runpp(self.net)

        # compute the next observation
        next_obs = self.observation_generator.generate_observation()

        # compute the next reward
        reward = self.reward_generator.generate_reward()

        return next_obs, reward, False, False, {}
    
    def _handle_action(self, action: np.ndarray):

        if len(action.shape) == 1:
            action = action.reshape(-1, 2)

        assert len(action) == len(self.ctrl_sgen_indices), "Action must have one entry per control gen"

        return action

    def _sample_scenario(self):
        self.scenario_id = self.np_random.integers(0, self.dataset.get_length())
        return self.dataset.get_item(self.scenario_id)
    
    def _apply_scenario(self, scenario, random_setpoint_reset):
        '''
        Applies the provided scenario to the network.
        '''

        # for now only looks at loads and sgens
        if 'load' in scenario:
            self.net['load'].loc[:, 'p_mw'] = scenario['load']['p_mw']
            self.net['load'].loc[:, 'q_mvar'] = scenario['load']['q_mvar']

        if 'sgen' in scenario:
            self.net['sgen'].loc[:, 'p_mw'] = scenario['sgen']['p_mw']
            self.net['sgen'].loc[:, 'q_mvar'] = scenario['sgen'].get('q_mvar', 0)
            self.net['sgen'].loc[:, 'max_p_mw'] = scenario['sgen']['max_p_mw']
            self.net['sgen'].loc[:, 'min_p_mw'] = scenario['sgen'].get('min_p_mw', 0)
            self.net['sgen'].loc[:, 'max_q_mvar'] = scenario['sgen']['max_q_mvar']

        if random_setpoint_reset:
            # Randomize the initial P,Q setpoint by rejection sampling.
            # WARNING: This can be arbitrarily slow and even fail in the worst case if the PQ area is very small compared to the square from -1 to 1
            # In most cases this should work just fine with a few iterations
            #setpoint = np.zeros(shape=(self.no_agents, 2))
            setpoints = []
            indices = []
            no_resampling_steps = 200
            for agent_idx in range(self.no_agents):
                step = 0
                while step < no_resampling_steps:
                    pq = self.np_random.uniform(low=-1, high=1, size=(2, ))
                    if self.pq_areas[agent_idx].is_inside(pq[0], pq[1]):
                        setpoints.append(pq)
                        indices.append(self.ctrl_sgen_indices[agent_idx])
                        break

                    step += 1
                
                if step == no_resampling_steps: # has not been successful
                    print('WARNING: rejection sampling did not converge. Could not sample a pq-setpoint on episode reset for agent.')

            setpoints = np.vstack(setpoints)

            # apply the sampled setpoints to the network
            self.net['sgen'].loc[indices, 'p_mw'] = self.net['sgen'].loc[indices, 'max_p_mw'] * setpoints[:, 0]
            self.net['sgen'].loc[indices, 'q_mvar'] = self.net['sgen'].loc[indices, 'max_q_mvar'] * setpoints[:, 1]

            # update the scenario
            scenario['sgen']['p_mw'] = np.array(self.net['sgen'].loc[:, 'p_mw'])
            scenario['sgen']['q_mvar'] = np.array(self.net['sgen'].loc[:, 'q_mvar'])

        # save the final scenario
        self.scenario = scenario


class DeltaStepVoltageControlEnv(VoltageControlEnv):
    '''
    A variant of the Voltage Control Environment. Instead of setting the desired setpoint,
    the agent specifies its desired absolute change of the current setpoint.
    '''
    def __init__(self, pp_net,
                 dataset: DataSet,
                 control_sgen_idx,
                 pq_areas: List[PQArea],
                 reward_generator : RewardGenerator,
                 observation_generator : ObservationGenerator,
                 random_setpoint_reset: bool = True,
                 delta_step: float = 0.1):
        '''
        Initializes the environment.
        Parameters (additional to base class):
            delta_step: The maximum possible absolute step size for active (p_mw) and reactive (q_mvar) power.
        '''
        super().__init__(pp_net=pp_net,
                         dataset=dataset,
                         control_sgen_idx=control_sgen_idx,
                         pq_areas=pq_areas,
                         reward_generator=reward_generator,
                         observation_generator=observation_generator,
                         random_setpoint_reset=random_setpoint_reset)
        
        # TODO: possible allow for different deltas depending on
        # p, q, bus.. if this is desirable
        self.delta_step = delta_step

    def step(self, action):
        '''
        Progresses the environment by one step. Computing the successor observation and emitted reward.
        Parameters:
            action: numpy array of shape (num_control_sgens, 2) or (num_control_sgens * 2,)
                    relative real and reactive power injection per controlled sgen
        Returns:
            next_obs, reward, terminated, truncated, info
        '''
        action = self._handle_action(action)

        # compute the desired injection change
        p_inj_delta = action[:, 0] * self.delta_step
        q_inj_delta = action[:, 1] * self.delta_step

        scaling = self.net.sgen.loc[self.ctrl_sgen_indices, 'scaling']  # scaling changes the step size 
        self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'] += p_inj_delta / scaling
        self.net.sgen.loc[self.ctrl_sgen_indices, 'q_mvar'] += q_inj_delta / scaling

        self._clip_action_to_feasible()

         # run power flow
        pp.runpp(self.net)

        # compute the next observation
        next_obs = self.observation_generator.generate_observation()

        # compute the next reward
        reward = self.reward_generator.generate_reward()

        return next_obs, reward, False, False, {}

    
    def _clip_action_to_feasible(self,):
        '''
        Makes sure that the set action (p_inj, q_inj) is inside the feasible region by clipping.
        '''

        # set desired active and reactive power injections rel * max + (1-rel) * min

        # convert the clip region to absolute values
        p_flexibilities = np.vstack([pq_area.total_p_flexibility() for pq_area in self.pq_areas])
        effective_min_p = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] * p_flexibilities[:, 0], dtype=np.float64)
        effective_max_p = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] * p_flexibilities[:, 1], dtype=np.float64)

        # Clip p to [p_min, p_max]
        self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'] = np.clip(np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'], dtype=np.float64),
                                                                    effective_min_p,
                                                                    effective_max_p)
        
        # convert the q-clip region to absolute values
        p_pu = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'p_mw'] / (self.net.sgen.loc[self.ctrl_sgen_indices, 'max_p_mw'] + 10**-7), dtype=np.float64)
        q_flexibilities = np.vstack([pq_area.q_flexibility([p_pu[idx]]) for idx, pq_area in enumerate(self.pq_areas)])
        effective_min_q = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_q_mvar'] * q_flexibilities[:, 0], dtype=np.float64)
        effective_max_q = np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'max_q_mvar'] * q_flexibilities[:, 1], dtype=np.float64)

        # Clip q to the corresponding values
        self.net.sgen.loc[self.ctrl_sgen_indices, 'q_mvar'] = np.clip(np.array(self.net.sgen.loc[self.ctrl_sgen_indices, 'q_mvar'], dtype=np.float64),
                                                                    effective_min_q,
                                                                    effective_max_q)

class MeanRewardWrapper(gym.RewardWrapper):
    '''
    Environment wrapper used to transform the multi-agent reward output to the mean scalar value.
    Can be used for (standard) centralized control approaches (and is conform with Gymnasium specification.)
    '''
    def __init__(self, env: VoltageControlEnv):
        super(MeanRewardWrapper, self).__init__(env)

    def reward(self, reward):
        return np.mean(list(reward.values()))

class FlattenObservationWrapper(gym.ObservationWrapper):
    '''
    Environment wrapper used to transform the nested multi-agent observation output to a one-dimensional numpy array by flattening.
    Can be used for (standard) centralized control approaches (and is conform with Gymnasium specification.)
    '''
    def __init__(self, env: VoltageControlEnv):
        super(FlattenObservationWrapper, self).__init__(env)
        self.single_agent_obs_size = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=np.tile(self.env.observation_space.low, env.no_agents),
                                                high=np.tile(self.env.observation_space.high, env.no_agents),
                                                shape=(self.env.observation_space.shape[0] * env.no_agents,))

    def observation(self, obs):
        new_obs = np.zeros(shape=(self.observation_space.shape))

        for idx, named_obs in enumerate(obs.values()):
            new_obs[idx*self.single_agent_obs_size : (idx+1)*self.single_agent_obs_size] = np.array(list(named_obs.values()))

        return new_obs


if __name__ == '__main__':

    from .env_utils import StandardRewardGenerator, StandardObservationGenerator, create_PQ_area
    
    sim_data = SimbenchDataSet("1-LV-rural1--0-sw")
    sgen_indices = list(range(len(sim_data.net.sgen)))
    pq_areas = [create_PQ_area('cone') for _ in sgen_indices]

    reward_gen = StandardRewardGenerator(sim_data.net, sgen_indices, min_volt=0.95, max_volt=1.05)
    obs_gen = StandardObservationGenerator(sim_data.net, sgen_indices)

    env = VoltageControlEnv(pp_net=sim_data.net,
                            dataset=sim_data,
                            control_sgen_idx=sgen_indices,
                            pq_areas=pq_areas,
                            reward_generator=reward_gen,
                            observation_generator=obs_gen,
                            #delta_step=0.001
                            )
    
    env = FlattenObservationWrapper(env)
    env = MeanRewardWrapper(env)

    print(env.observation_space.shape)

    #print("Before Application: ", env.net['load'][['p_mw', 'q_mvar']])
    obs, _ = env.reset(options={'scenario_id': 14100})
    #print("Scenario: ", env.scenario['load'])
    #print("After Application: ", env.net['load'][['p_mw', 'q_mvar']])

    print("Before step:", obs)
    print("Before Step: ", env.net['sgen'][['p_mw', 'q_mvar', 'max_q_mvar', 'max_p_mw']])

    action = np.ones((len(sgen_indices), 2)) * -1
    # action[:, 1] = action[:, 1] * -1
    action[:, 0] = action[:, 0] * 0.1961722488

    print(action)

    obs, reward, _, _, _ = env.step(action)

    print("After step: ", obs)
    print("After Step: ", env.net['sgen'][['p_mw', 'q_mvar', 'max_q_mvar', 'max_p_mw']])

    print("Reward:", reward)