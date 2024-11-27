from typing import List

import pandapower as pp
import pandapower.plotting as plot
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .dataset import DataSet, SimbenchDataSet
from .env_utils import ObservationGenerator, RewardGenerator, PQArea, ScenarioManager

class VoltageControlEnv(gym.Env):
    '''
    The Base Class of the Voltage Control Env implementing the Gymnasium Env interface.
    The environment is formulated in a more general multi-agent style and deviates from the standard Gymnasium specification.
    E.g. it returns a dictionary of rewards (one for every agent) instead of a single scalar value.

    In this environment the agent specifies his action by directly setting a desired setpoint in the feasible PQ-area.
    '''

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
        }
    
    def __init__(self, 
                 scenario_manager : ScenarioManager,
                 reward_generator : RewardGenerator,
                 observation_generator : ObservationGenerator,
                 random_setpoint_reset: bool = True,
                 step_delay: float = 0.0):
        '''
        Initializes the environment.
        Parameters:
            scenario_manager: Scenario Manager used for managing the scenario's data (net, dataset, controllable indices etc.)
            reward_generator: Reward Generator used for generating rewards.
            observation_generator: Observation Generator used for generating environment observations.
            random_setpoint_reset: Whether the setpoint is randomized (among feasible pq setpoints) upon reset or not.
            step_delay: Floating number in the range of [0,1] specifiying the speed of convergence to a desired setpoint.
                        For delay = 0 the setpoint is reached in one step.
        '''
        self.sm = scenario_manager
        self.reward_generator = reward_generator
        self.observation_generator = observation_generator
        self.random_setpoint_reset = random_setpoint_reset
        self.step_delay = step_delay
        
        self.scenario_id = None
        self.scenario = None

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.sm.no_agents, 2))
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
                scenario = self.sm.dataset.get_item(options['scenario_id'])

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
        pp.runpp(self.sm.net)

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
        p_flexibilities = np.vstack([pq_area.total_p_flexibility() for pq_area in self.sm.pq_areas])
        effective_min_p = np.maximum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'min_p_mw'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * p_flexibilities[:, 0]).astype(np.float64)
        effective_max_p = np.minimum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'max_p_mw'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * p_flexibilities[:, 1]).astype(np.float64)

        p_inj = rel_p_action * effective_max_p + (1 - rel_p_action) * effective_min_p

        p_pu = p_inj / (self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'] + 10**-7)
        q_flexibilities = np.vstack([pq_area.q_flexibility([p_pu[idx]]) for idx, pq_area in enumerate(self.sm.pq_areas)])
        effective_min_q = np.maximum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'min_q_mvar'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * q_flexibilities[:, 0]).astype(np.float64)
        effective_max_q = np.minimum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'max_q_mvar'].to_numpy(), 
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * q_flexibilities[:, 1]).astype(np.float64)
        
        q_inj = (action[:, 1] + 1) / 2 * effective_max_q + (1 - (action[:, 1] +  1) / 2) * effective_min_q

        # apply to net considering the delay
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'] = self.step_delay * self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'] + (1 - self.step_delay) * p_inj
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'q_mvar'] = self.step_delay * self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'q_mvar'] + (1 - self.step_delay) * q_inj

        # run power flow
        pp.runpp(self.sm.net)

        # compute the next observation
        next_obs = self.observation_generator.generate_observation()

        # compute the next reward
        reward = self.reward_generator.generate_reward()

        return next_obs, reward, False, False, {}
    
    def render(self,):

        NODE_SIZE = 0.07
        MAX_SIZE = 0.5
        RING_SIZE = 0.005
        max_sn = np.max(self.sm.net.sgen['sn_mva'])

        # retrieve the busses which do not have any flexibilities whatsoever
        flex_bus_idx = self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'bus'].to_numpy()
        non_flex_bus_idx = np.array([bus for bus in self.sm.net.bus.index if bus not in flex_bus_idx])

        lc = plot.create_line_collection(self.sm.net, self.sm.net.line.index, color="grey", zorder=1, use_bus_geodata=True) #create lines
        tc = plot.create_trafo_collection(self.sm.net, self.sm.net.trafo.index, size=NODE_SIZE, color="grey", zorder=1) # create trafos
        bc_inactive = plot.create_bus_collection(self.sm.net, non_flex_bus_idx, size=NODE_SIZE, color='grey', zorder=1) # create inactive buses

        # for every active bus compute exactly the size of outer ring, inner circle and color of circle
        bcs_active = []
        cmap = plt.cm.coolwarm
        norm = mcolors.Normalize(vmin=0, vmax=1)

        for i, (id, bus_id) in enumerate(zip(self.sm.ctrl_sgen_indices, flex_bus_idx)):
            # outer ring
            outer_ring_size = (self.sm.net.sgen.loc[id, 'max_p_mw'] / (max_sn + 1e-7)) * MAX_SIZE
            bcs_active.append(plot.create_bus_collection(self.sm.net, [bus_id], size=outer_ring_size, color='black', zorder=1))
            bcs_active.append(plot.create_bus_collection(self.sm.net, [bus_id], size=outer_ring_size-RING_SIZE, color='white', zorder=2))

            # inner_circle
            # compute the size of the inner circle based on the p injection
            inner_circle_size = (self.sm.net.sgen.loc[id, 'p_mw'] / (self.sm.net.sgen.loc[id, 'max_p_mw'] + 1e-7)) * outer_ring_size
            # compute the color of the inner circle based on the q injection
            rel_p_inj = self.sm.net.sgen.loc[id, 'p_mw'] / (self.sm.net.sgen.loc[id, 'sn_mva'] + 1e-7)
            q_flex = (self.sm.pq_areas[i].q_flexibility([rel_p_inj]) * self.sm.net.sgen.loc[id, 'sn_mva'])[0]
            q_rel = (self.sm.net.sgen.loc[id, 'q_mvar'] - q_flex[0]) / (q_flex[1]-q_flex[0] + 1e-7)  # in the range of 0 to 1

            # Get the color from the colormap
            color = cmap(norm(q_rel))
            # color = 'blue'
                
            bcs_active.append(plot.create_bus_collection(self.sm.net, [bus_id], size=inner_circle_size, color=color, zorder=3))

        # create the network state plot
        #cmap_lc_list_load=[(20, "green"), (50, "yellow"), (80, "red")]
        #cmap_lc_load, norm_lc_load = plot.cmap_continuous(cmap_lc_list_load)

        #lc_load = plot.create_line_collection(self.sm.net, self.sm.net.line.index, zorder=1, cmap=cmap_lc_load, norm=norm_lc_load, use_bus_geodata=True, plot_colormap=False)
        lc_load = plot.create_line_collection(self.sm.net, self.sm.net.line.index, zorder=1, color='grey', use_bus_geodata=True, plot_colormap=False)

        # cmap_bc_list_load=[(0.93, "blue"),(1.0, "green"), (1.07, "red")]
        # cmap_bc_load, norm_bc_load = plot.cmap_continuous(cmap_bc_list_load)
        cmap_bc_load = plt.cm.viridis
        norm_bc_load = mcolors.Normalize(vmin=0.93, vmax=1.07)

        bc_load = plot.create_bus_collection(self.sm.net, self.sm.net.bus.index, zorder=2, cmap=cmap_bc_load, norm=norm_bc_load, size=NODE_SIZE, plot_colormap=False)

        tc_load = plot.create_trafo_collection(self.sm.net, self.sm.net.trafo.index, zorder=1, color='grey', size=NODE_SIZE)

        # Draw
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'hspace': 0.00})
        plot.draw_collections([lc_load, bc_load, tc_load], ax=axs[0])
        plot.draw_collections([lc, tc, bc_inactive] + bcs_active, ax=axs[1])
        
        # Add a colorbar for load map
        mapp_load = plt.cm.ScalarMappable(cmap=cmap_bc_load, norm=norm_bc_load)  # Create a scalar mappable
        mapp_load.set_array([])  # Required for colorbar
        cbar_load = fig.colorbar(mapp_load, ax=axs[0], orientation='vertical', label="Bus Voltage [pu]", shrink=0.75)

        # Design
        cbar_load.outline.set_visible(False)

        # Add a colorbar
        mapp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Create a scalar mappable
        mapp.set_array([])  # Required for colorbar
        cbar = fig.colorbar(mapp, ax=axs[1], orientation='vertical', label='Q-injection', shrink=0.75)  # Add colorbar

        # Set custom ticks and labels
        tick_positions = [0, 1]  # Positions where you want ticks
        tick_labels = ['Min','Max']  # Custom labels
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

        # Design
        cbar.outline.set_visible(False)

        # plt.tight_layout(pad=0)

        # convert to RGB array
        # Draw the figure on a canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Extract the RGBA buffer
        buf = canvas.buffer_rgba()
        
        # Convert to a NumPy array and drop the alpha channel
        rgb_array = np.asarray(buf, dtype=np.uint8)[:, :, :3]

        # cleaning up
        plt.close()
        
        return rgb_array

    
    def _handle_action(self, action: np.ndarray):

        if len(action.shape) == 1:
            action = action.reshape(-1, 2)

        assert len(action) == len(self.sm.ctrl_sgen_indices), "Action must have one entry per control gen"

        return action

    def _sample_scenario(self):
        self.scenario_id = self.np_random.integers(0, self.sm.dataset.get_length())
        return self.sm.dataset.get_item(self.scenario_id)
    
    def _apply_scenario(self, scenario, random_setpoint_reset):
        '''
        Applies the provided scenario to the network.
        '''

        # for now only looks at loads and sgens
        if 'load' in scenario:
            self.sm.net['load'].loc[:, 'p_mw'] = scenario['load']['p_mw']
            self.sm.net['load'].loc[:, 'q_mvar'] = scenario['load']['q_mvar']

        if 'sgen' in scenario:
            self.sm.net['sgen'].loc[:, 'p_mw'] = scenario['sgen']['p_mw']
            self.sm.net['sgen'].loc[:, 'q_mvar'] = scenario['sgen'].get('q_mvar', 0)
            self.sm.net['sgen'].loc[:, 'max_p_mw'] = scenario['sgen']['max_p_mw']
            self.sm.net['sgen'].loc[:, 'min_p_mw'] = scenario['sgen'].get('min_p_mw', 0)

            # Note: These values for the q_bounds are general maxima which do not hold for every active power p
            # Set for the sake of completeness. For the exact bound the PQ-areas have to be considered
            self.sm.net['sgen'].loc[:, 'max_q_mvar'] = self.sm.net['sgen'].loc[:, 'sn_mva']  # 1pu is the s_max bound
            self.sm.net['sgen'].loc[:, 'min_q_mvar'] = -self.sm.net['sgen'].loc[:, 'sn_mva']

        if random_setpoint_reset:
            # Randomize the initial P,Q setpoint uniformly by rejection sampling.
            # WARNING: This can be arbitrarily slow and even fail in the worst case if the PQ area is very small compared to
            # the action space constrained by max/min p_mw and max/min q_mvar.
            # In most relevant cases this should work just fine with a few iterations
            setpoints = []
            indices = []
            no_resampling_steps = 100
            for agent_idx in range(self.sm.no_agents):
                step = 0
                min_p, max_p = self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices[agent_idx], 'min_p_mw'], self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices[agent_idx], 'max_p_mw']
                min_q, max_q = self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices[agent_idx], 'min_q_mvar'], self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices[agent_idx], 'max_q_mvar']
                max_s = self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices[agent_idx], 'sn_mva']
                while step < no_resampling_steps:
                    # sample point 
                    p = self.np_random.uniform(low=min_p, high=max_p, size=1).item()
                    q = self.np_random.uniform(low=min_q, high=max_q, size=1).item()

                    if self.sm.pq_areas[agent_idx].is_inside(p/(max_s + 1e-7), q/(max_s + 1e-7)):  # if inside the region
                        setpoints.append((p,q))
                        indices.append(self.sm.ctrl_sgen_indices[agent_idx])
                        break

                    step += 1
                
                # if step == no_resampling_steps: # has not been successful
                #     print('WARNING: rejection sampling did not converge. Could not sample a pq-setpoint on episode reset for agent.')

            setpoints = np.array(setpoints)

            # apply the sampled setpoints to the network
            if len(indices) > 0:
                self.sm.net['sgen'].loc[indices, 'p_mw'] = setpoints[:, 0]
                self.sm.net['sgen'].loc[indices, 'q_mvar'] = setpoints[:, 1]

            # update the scenario
            scenario['sgen']['p_mw'] = np.array(self.sm.net['sgen'].loc[:, 'p_mw'])
            scenario['sgen']['q_mvar'] = np.array(self.sm.net['sgen'].loc[:, 'q_mvar'])

        # save the final scenario
        self.scenario = scenario


class DeltaStepVoltageControlEnv(VoltageControlEnv):
    '''
    A variant of the Voltage Control Environment. Instead of setting the desired setpoint,
    the agent specifies its desired absolute change of the current setpoint.
    '''
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
        }


    def __init__(self,
                 scenario_manager: ScenarioManager,
                 reward_generator : RewardGenerator,
                 observation_generator : ObservationGenerator,
                 random_setpoint_reset: bool = True,
                 delta_step: float = 0.1):
        '''
        Initializes the environment.
        Parameters (additional to base class):
            delta_step: The maximum possible absolute step size for active (p_mw) and reactive (q_mvar) power.
        '''
        super().__init__(scenario_manager=scenario_manager,
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

        scaling = self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'scaling']  # scaling changes the step size 
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'] += p_inj_delta / scaling
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'q_mvar'] += q_inj_delta / scaling

        self._clip_action_to_feasible()

         # run power flow
        pp.runpp(self.sm.net)

        # compute the next observation
        next_obs = self.observation_generator.generate_observation()

        # compute the next reward
        reward = self.reward_generator.generate_reward()

        return next_obs, reward, False, False, {}

    
    def _clip_action_to_feasible(self,):
        '''
        Makes sure that the set action (p_inj, q_inj) is inside the feasible region by clipping.
        '''

        # convert the clip region to absolute values
        p_flexibilities = np.vstack([pq_area.total_p_flexibility() for pq_area in self.sm.pq_areas])
        effective_min_p = np.maximum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'min_p_mw'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * p_flexibilities[:, 0]).astype(np.float64)
        effective_max_p = np.minimum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'max_p_mw'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * p_flexibilities[:, 1]).astype(np.float64)

        # Clip p to [p_min, p_max]
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'] = np.clip(np.array(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'], dtype=np.float64),
                                                                          effective_min_p,
                                                                          effective_max_p)
        
        # convert the q-clip region to absolute values
        p_pu = np.array(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'p_mw'] / (self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'] + 10**-7), dtype=np.float64)
        q_flexibilities = np.vstack([pq_area.q_flexibility([p_pu[idx]]) for idx, pq_area in enumerate(self.sm.pq_areas)])
        effective_min_q = np.maximum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'min_q_mvar'].to_numpy(),
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * q_flexibilities[:, 0]).astype(np.float64)
        effective_max_q = np.minimum(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'max_q_mvar'].to_numpy(), 
                                     self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'sn_mva'].to_numpy() * q_flexibilities[:, 1]).astype(np.float64)

        # Clip q to the corresponding values
        self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'q_mvar'] = np.clip(np.array(self.sm.net.sgen.loc[self.sm.ctrl_sgen_indices, 'q_mvar'], dtype=np.float64),
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
        self.observation_space = gym.spaces.Box(low=np.tile(self.env.observation_space.low, env.sm.no_agents),
                                                high=np.tile(self.env.observation_space.high, env.sm.no_agents),
                                                shape=(self.env.observation_space.shape[0] * env.sm.no_agents,))

    def observation(self, obs):
        new_obs = np.zeros(shape=(self.observation_space.shape))

        for idx, named_obs in enumerate(obs.values()):
            new_obs[idx*self.single_agent_obs_size : (idx+1)*self.single_agent_obs_size] = np.array(list(named_obs.values()))

        return new_obs