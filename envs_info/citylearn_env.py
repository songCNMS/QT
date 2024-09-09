import gymnasium as gym
import os
import numpy as np
import json
import sys
from citylearn.citylearn import EvaluationCondition, CityLearnEnv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime




def plot_env(
    active_obs,
    statelist,
    active_actions,
    actionlist,
    rewardlist,
    baseline_statelist,
    loc,
    reward_components_list=None,
):
    for i, obs_name in enumerate(active_obs):
        y = [s[obs_name] for s in statelist]
        sns.lineplot(x=list(range(len(statelist))), y=y)
        if obs_name == "indoor_dry_bulb_temperature":
            y = [s[obs_name + "_set_point"] for s in statelist]
            sns.lineplot(x=list(range(len(statelist))), y=y)

        plt.savefig(f"{loc}/obs_{obs_name}.png")
        plt.close()
    for i, action_name in enumerate(active_actions):
        y = [s[i] for s in actionlist]
        sns.lineplot(x=list(range(len(actionlist))), y=y)
        plt.savefig(f"{loc}/action_{action_name}.png")
        plt.close()

    col = "net_electricity_consumption"
    y = [s[col] for s in statelist]
    sns.lineplot(x=list(range(len(statelist))), y=y)
    y = [s[col] for s in baseline_statelist]
    sns.lineplot(x=list(range(len(statelist))), y=y)
    plt.savefig(f"{loc}/obs_{col}_with_baseline.png")
    plt.close()

    col = "net_electricity_consumption_emission"
    y = [s["net_electricity_consumption"] * s["carbon_intensity"] for s in statelist]
    sns.lineplot(x=list(range(len(statelist))), y=y)
    y = [s[col] for s in baseline_statelist]
    sns.lineplot(x=list(range(len(statelist))), y=y)
    plt.savefig(f"{loc}/obs_{col}_with_baseline.png")
    plt.close()
    
    col = "community_net_electricity_consumption"
    if col in statelist[0]:
        y = [s[col] for s in statelist]
        sns.lineplot(x=list(range(len(statelist))), y=y)
        y = [s["community_net_electricity_consumption_baseline"] for s in statelist]
        sns.lineplot(x=list(range(len(statelist))), y=y)
        plt.savefig(f"{loc}/obs_{col}.png")
        plt.close()

    sns.lineplot(x=list(range(len(rewardlist))), y=rewardlist)
    plt.savefig(f"{loc}/step_reward.png")
    plt.close()

    if reward_components_list is not None:
        df = pd.DataFrame(reward_components_list)
        sns.lineplot(data=df)
        plt.savefig(f"{loc}/reward_components.png")
        plt.close()
    plt.cla()


def to_plot_env(eval_env, dir_loc):
    for bldg_idx, building in enumerate(eval_env.city_learn_env.buildings):
        active_obs = building.active_observations
        active_actions = building.active_actions
        (
            statelist,
            actionlist,
            rewardlist,
            reward_component_list,
            baseline_statelist,
        ) = (
            eval_env.statelist[bldg_idx],
            eval_env.actionlist[bldg_idx],
            eval_env.rewardlist[bldg_idx],
            eval_env.reward_components_list[bldg_idx],
            eval_env.baseline_statelist[bldg_idx],
        )
        plot_loc = os.path.join(dir_loc, f"bldg_{bldg_idx}")
        os.makedirs(plot_loc, exist_ok=True)
        plot_env(
            active_obs,
            statelist,
            active_actions,
            actionlist,
            rewardlist,
            baseline_statelist,
            plot_loc,
            reward_components_list=reward_component_list,
        )



def get_comfort_reward(prev_reward_obss, cur_reward_obss, baseline_rewards, bldg_idx):
    unmet = 0.0
    heating = cur_reward_obss[bldg_idx].get("heating_demand", 0) > cur_reward_obss[
        bldg_idx
    ].get("cooling_demand", 0)
    if heating:
        lb, ub = 0.0, 0.99
    else:
        lb, ub = 0.0, 0.99
    if cur_reward_obss[bldg_idx]["occupant_count"] > 0:
        temp_gap = cur_reward_obss[bldg_idx]["indoor_dry_bulb_temperature"] - cur_reward_obss[bldg_idx]["indoor_dry_bulb_temperature_set_point"]
        temp_gap = min(5.0, max(-5.0, temp_gap))
        unmet = abs(temp_gap)
        if -1.0 < temp_gap <= 0.0: unmet = 0.5
        elif 0.0 <= temp_gap < 1.0: unmet = 0.0
        else: unmet += 1.0
    return 6.0 - unmet


def get_emission_reward(prev_reward_obss, cur_reward_obss, baseline_rewards, bldg_idx):
    carbon_emission = (
        cur_reward_obss[bldg_idx]["carbon_intensity"]
        * cur_reward_obss[bldg_idx]["net_electricity_consumption"]
    )
    carbon_emission /= (1.0 if baseline_rewards[bldg_idx]["net_electricity_consumption_emission"] <= 0.0 else baseline_rewards[bldg_idx]["net_electricity_consumption_emission"])
    carbon_emission = min(2.0, max(0.0, carbon_emission))
    return (10.0 - carbon_emission*5.0) / 5.0


def get_ramping_reward(
    prev_reward_obss, cur_reward_obss, baseline_rewards, hist_consumptions, bldg_idx
):
    community_electricity_consumption = cur_reward_obss[bldg_idx]["net_electricity_consumption"]
    baseline_community_electricity_consumption = baseline_rewards[bldg_idx]["net_electricity_consumption"]
    prev_community_electricity_consumption = prev_reward_obss[bldg_idx]["net_electricity_consumption"]
    consumption_ramp = ((abs(community_electricity_consumption-baseline_community_electricity_consumption) 
              + abs(community_electricity_consumption-prev_community_electricity_consumption)) 
              / (1.0 if baseline_community_electricity_consumption <= 0.0 else baseline_community_electricity_consumption))
    consumption_ramp = min(1.0, max(0.0, consumption_ramp))
    return (10.0 - consumption_ramp*5.0) / 5.0


class GPTCityLearnWrapper(gym.Env):
    def __init__(self, env, seed, agents, center=False, normalized=False):
        self.hist_net_electricity_consumption_num = 24
        self.city_learn_env = env  # CityLearnEnv(schema, random_seed=seed)
        self.episode_len = 719
        self.render_freq = 50
        self.normalized = normalized
        self.central_agent = center
        self.num_buildings = len(self.city_learn_env.buildings)
        self.all_agents = ["building%i"%(i+1) for i in range(self.num_buildings)]
        self.hist_net_electricity_consumption = [
            0
        ] * self.hist_net_electricity_consumption_num
        
        self.obss = None
        self.selected_agents = agents
        
        
        

        self.observation_names = env.observation_names
        if self.central_agent:
            self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.selected_agents)*self.city_learn_env.buildings[0].observation_space.shape[0],))
            self.action_space = gym.spaces.Box(0, 1, shape=(len(self.selected_agents)*self.city_learn_env.buildings[0].action_space.shape[0],))
        else:
            self.observation_space = {f"building{i+1}": self.city_learn_env.buildings[i].observation_space for i in range(self.num_buildings)}
            self.action_space = {f"building{i+1}": self.city_learn_env.buildings[i].action_space for i in range(self.num_buildings)}
        self.observation_dict = {f"building{i+1}": self.city_learn_env.buildings[i].get_metadata()["observation_metadata"] for i in range(self.num_buildings)}
        self.time_steps = env.time_steps

        baseline_condition = (
            EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
        )
        baseline_condition = (
            EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
        )
        self.baseline_net_electricity_consumption = lambda x: getattr(
            x, f"net_electricity_consumption{baseline_condition.value}"
        )
        self.baseline_net_electricity_consumption_cost = lambda x: getattr(
            x, f"net_electricity_consumption_cost{baseline_condition.value}"
        )
        self.baseline_net_electricity_consumption_emission = lambda x: getattr(
            x, f"net_electricity_consumption_emission{baseline_condition.value}"
        )

        self.reward_components_list = [
            {"comfort": [], "emission": [], "ramping": []}
            for i in range(self.num_buildings)
        ]

        self.building_info = [{} for _ in range(self.num_buildings)]
        for i, b in enumerate(self.city_learn_env.buildings):
            for attr in [
                "cooling_device",
                "heating_device",
                "dhw_storage",
                "electrical_storage",
                "cooling_storage",
                "heating_storage",
            ]:
                val_name = "nominal_power" if attr.find("device") >= 0 else "capacity"
                attr_name = f"_Building__{attr}"
                self.building_info[i][attr] = getattr(getattr(b, attr_name), val_name)
                
        self.statelist = [[] for i in range(self.num_buildings)]
        self.actionlist = [[] for i in range(self.num_buildings)]
        self.rewardlist = [[] for i in range(self.num_buildings)]
        self.sub_rewardlist = [[] for i in range(self.num_buildings)]
        self.baseline_statelist = [[] for i in range(self.num_buildings)]

    def seed(self, _seed):
        pass

    def get_baseline(self, building):
        return {
            "net_electricity_consumption": self.baseline_net_electricity_consumption(
                building
            )[-1],
            "net_electricity_consumption_cost": self.baseline_net_electricity_consumption_cost(
                building
            )[
                -1
            ],
            "net_electricity_consumption_emission": self.baseline_net_electricity_consumption_emission(
                building
            )[
                -1
            ],
        }

    def state_normalization(self, states):
        return [
            (s - self.city_learn_env.buildings[i].observation_space.low)
            / (
                self.city_learn_env.buildings[i].observation_space.high
                - self.city_learn_env.buildings[i].observation_space.low
            )
            for i, s in enumerate(states)
        ]

    def state_denormalization(self, states):
        return [
            s
            * (
                self.city_learn_env.buildings[i].observation_space.high
                - self.city_learn_env.buildings[i].observation_space.low
            )
            + self.city_learn_env.buildings[i].observation_space.low
            for i, s in enumerate(states)
        ]
        
    def get_normalized_score(self, score):
        # return (score-300.0)/(500.0-300.0)
        metrics_df = self.city_learn_env.evaluate_citylearn_challenge()
        print(metrics_df)
        dir_loc = f"logs/citylearn/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(dir_loc, exist_ok=True)
        with open(f"{dir_loc}/metric.json", "w") as f:
            json.dump(str(metrics_df), f)
        to_plot_env(self, dir_loc)
        return 1.0 - metrics_df["average_score"]["value"]

    def step(self, actions):
        prev_reward_obss = [
            b.observations(
                include_all=True,
                normalize=False,
                periodic_normalization=False,
                # reward_calc=False,
            )
            for b in self.city_learn_env.buildings
        ]
        if self.central_agent:
            # env_actions = [self.city_learn_env.buildings[i].action_space.sample() for i in range(self.num_buildings)]
            env_actions = np.zeros((self.num_buildings, self.city_learn_env.buildings[0].action_space.shape[0]))
            ppo_actions = np.array(actions).reshape(len(self.selected_agents), -1)
            for i, agent in enumerate(self.selected_agents):
                env_actions[self.all_agents.index(agent)] = ppo_actions[i]
        else:
            env_actions = [act for act in actions.values()]
        
        
        for i in range(self.num_buildings):
            building = self.city_learn_env.buildings[i]
            for j, key in enumerate(building.active_actions):
                env_actions[i][j] = min(building.action_space.high[j]-1e-3, max(env_actions[i][j], building.action_space.low[j]+1e-3))
        
        next_state, _, done, _ = self.city_learn_env.step(env_actions)
        actions = env_actions
        cur_reward_obss = [
            b.observations(
                include_all=True,
                normalize=False,
                periodic_normalization=False,
                # reward_calc=False,
            )
            for b in self.city_learn_env.buildings
        ]
        baseline_reward_obss = [
            self.get_baseline(b) for b in self.city_learn_env.buildings
        ]
        self.hist_net_electricity_consumption.append(
            np.sum([r["net_electricity_consumption"] for r in cur_reward_obss])
        )
        reward_components = [
            (
                get_comfort_reward(
                    prev_reward_obss, cur_reward_obss, baseline_reward_obss, bldg_idx
                ),
                get_emission_reward(
                    prev_reward_obss, cur_reward_obss, baseline_reward_obss, bldg_idx
                ),
                get_ramping_reward(
                    prev_reward_obss,
                    cur_reward_obss,
                    baseline_reward_obss,
                    self.hist_net_electricity_consumption,
                    bldg_idx
                ),
            )
            for bldg_idx in range(len(self.city_learn_env.buildings))
        ]
        for bldg_idx, (r1, r2, r3) in enumerate(reward_components):
            self.reward_components_list[bldg_idx]["comfort"].append(r1)
            self.reward_components_list[bldg_idx]["emission"].append(r2)
            self.reward_components_list[bldg_idx]["ramping"].append(r3)

        self.reward = [10.0 + np.sum(r) for r in reward_components]

        self.state = next_state.copy()
        if self.normalized:
            self.state = self.state_normalization(self.state)
        self.obss = []
        for bldg_idx in range(self.num_buildings):
            r_obs = cur_reward_obss[bldg_idx]
            r_obs["community_net_electricity_consumption"] = np.sum([r["net_electricity_consumption"] for r in cur_reward_obss])
            r_obs["community_net_electricity_consumption_baseline"] = np.sum([r["net_electricity_consumption"] for r in baseline_reward_obss])
            self.obss.append(r_obs)
            self.statelist[bldg_idx].append(r_obs)
            self.actionlist[bldg_idx].append(actions[bldg_idx])
            self.rewardlist[bldg_idx].append(self.reward[bldg_idx])
            self.sub_rewardlist[bldg_idx].append(self.get_sub_rewards(bldg_idx))
            self.baseline_statelist[bldg_idx].append(
                self.get_baseline(self.city_learn_env.buildings[bldg_idx])
            )
        
        if self.central_agent:
            return (np.concatenate([self.state[self.all_agents.index(agent)] for agent in self.selected_agents]),
                    np.sum([self.reward[self.all_agents.index(agent)] for agent in self.selected_agents]),
                    done,
                    # done,
                    {}
                )
        else:
            return ({f"building{i+1}": self.state[i] for i in range(self.num_buildings)},
                    {f"building{i+1}": self.reward[i] for i in range(self.num_buildings)}, 
                    {f"building{i+1}": done for i in range(self.num_buildings)}, 
                    # {f"building{i+1}": done for i in range(self.num_buildings)}, 
                    {f"building{i+1}": {} for i in range(self.num_buildings)})


    def get_sub_rewards(self, bldg_idx):
        return {
            key: self.reward_components_list[bldg_idx][key][-1]
            for key in ["comfort", "emission", "ramping"]
        }

    def reset(self, seed=0):
        self.reward_components_list = [
            {"comfort": [], "emission": [], "ramping": []}
            for i in range(self.num_buildings)
        ]
        self.statelist = [[] for i in range(self.num_buildings)]
        self.actionlist = [[] for i in range(self.num_buildings)]
        self.rewardlist = [[] for i in range(self.num_buildings)]
        self.sub_rewardlist = [[] for i in range(self.num_buildings)]
        self.baseline_statelist = [[] for i in range(self.num_buildings)]
        next_state = self.city_learn_env.reset()
        self.state = next_state.copy()
        if self.normalized:
            self.state = self.state_normalization(self.state)
        obss = [b.observations(
                include_all=True,
                normalize=False,
                periodic_normalization=False,
                # reward_calc=False,
            ) for b in self.city_learn_env.buildings]
        baseline_reward_obss = [
            self.get_baseline(b) for b in self.city_learn_env.buildings
        ]
        self.obss = []
        for bldg_idx in range(self.num_buildings):
            r_obs = obss[bldg_idx]
            r_obs["community_net_electricity_consumption"] = np.sum([r["net_electricity_consumption"] for r in obss])
            r_obs["community_net_electricity_consumption_baseline"] = np.sum([r["net_electricity_consumption"] for r in baseline_reward_obss])
            self.obss.append(r_obs)
            self.statelist[bldg_idx].append(r_obs)
            self.baseline_statelist[bldg_idx].append(
                self.get_baseline(self.city_learn_env.buildings[bldg_idx])
            )
        if self.central_agent:
            return np.concatenate([self.state[self.all_agents.index(agent)] for agent in self.selected_agents])#, {}
        else:
            return {f"building{i+1}": self.state[i] for i in range(self.num_buildings)}
            # return ({f"building{i+1}": self.state[i] for i in range(self.num_buildings)},
            #         {f"building{i+1}": {} for i in range(self.num_buildings)})
    
    def close(self):
        pass


def get_env(center=False, env_name='citylearn_challenge_2023_phase_2_local_evaluation', agents=["building1", "building2", "building3"], seed=0):
    num_episodes = 719
    ori_env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=False, random_seed=seed)
    seed = np.random.randint(num_episodes)
    env = GPTCityLearnWrapper(ori_env, seed, agents, center=center, normalized=True)
    return env



def get_building_obs_desc(o, o_dict, observation_names, agent):
    # print(observation_names, o, list(o_dict.keys()))
    context = f"The following data describes latest status of building {agent}: \n"
    context += f"Climate conditions: \n "
    for col, desc, unit in [
        (
            "indoor_dry_bulb_temperature",
            "indoor dry bulb temperature",
            "Celsius degrees",
        ),
        ("indoor_relative_humidity", "indoor relative humidity", "percent"),
        (
            "indoor_dry_bulb_temperature_set_point",
            "indoor dry bulb temperature set point",
            "Celsius degrees",
        ),
    ]:
        if o_dict[col]:
            context += f"{desc}: {o[observation_names.index(col)]} {unit} \n"
    storages = [
        (
            "cooling_storage_soc",
            "State of the charge (SOC) of the cooling storage from 0 (no energy stored) to 1 (at full capacity)",
            "kWh/kWhcapacity",
        ),
        (
            "heating_storage_soc",
            "State of the charge (SOC) of the heating storage from 0 (no energy stored) to 1 (at full capacity)",
            "kWh/kWhcapacity",
        ),
        (
            "electrical_storage_soc",
            "State of the charge (SOC) of the electrical storage from 0 (no energy stored) to 1 (at full capacity)",
            "kWh/kWhcapacity",
        ),
        (
            "dhw_storage_soc",
            "State of the charge (SOC) of the domestic hot water storage from 0 (no energy stored) to 1 (at full capacity)",
            "kWh/kWhcapacity",
        ),
    ]
    if np.any([o_dict[col] for col, _, _ in storages]):
        context += f"Energy storages in this building and their state of charge: \n "
        for col, desc, unit in storages:
            if o_dict[col]:
                context += f"{desc}: {o[observation_names.index(col)]} {unit} \n"

    if o_dict["solar_generation"]:
        context += f"The building is equipped with PV generator and generating {o[observation_names.index('solar_generation')]} kWh in the current period. "
    context += "Energy demands and electriciy consumption of the building:\n"
    for col, desc, unit in [
        (
            "non_shiftable_load",
            "Total building non-shiftable plug and equipment loads",
            "kWh",
        ),
        (
            "net_electricity_consumption",
            "Total building electricity consumption",
            "kWh",
        ),
        (
            "cooling_demand",
            "Cooling energy demand that shall be fulfilled by cooling device and/or cooling storage for space cooling",
            "kWh",
        ),
        (
            "heating_demand",
            "Heating energy demand that shall be fulfilled by heating device and/or heating storage for space heating",
            "kWh",
        ),
        (
            "dhw_demand",
            "Heating energy demand that shall be fulfilled by domestic hot water device and domestic hot water storage for DHW heating",
            "kWh",
        ),
        (
            "cooling_electricity_consumption",
            "net electricity consumption in meeting cooling demand and cooling storage demand",
            "kWh",
        ),
        (
            "heating_electricity_consumption",
            "net electricity consumption in meeting heating demand and heating storage demand",
            "kWh",
        ),
        (
            "dhw_electricity_consumption",
            "net electricity consumption in meeting domestic hot water demand and storage demand",
            "kWh",
        ),
    ]:
        if o_dict[col]:
            context += f"{desc}: {o[observation_names.index(col)]} {unit} \n"

    if o_dict["occupant_count"]:
        if o[observation_names.index("occupant_count")] == 0:
            context += "Currently, no accupant is in the building, so you can ignore thermal comfort requirements and use energy as less as possible. "
        else:
            context += f"Currently, there are occupants in the building, so you shall take the following thermal comfort requirement into consideration: the indoor dry bulb temperature shall be higher than {o[observation_names.index('indoor_dry_bulb_temperature_set_point')]-1.0} while lower than {o[observation_names.index('indoor_dry_bulb_temperature_set_point')]+1.0} Celsius degrees. "
    return context


def get_state_description(env, obss, agents=None):
    selected_agents = agents if agents else env.all_agents
    context = ""
    for agent in selected_agents:
        obs = obss[agent]
        obs_dict = env.observation_dict[agent]
        context += get_building_obs_desc(obs, obs_dict, env.observation_names[env.all_agents.index(agent)], agent)
        context += "\n"
    return context


def get_task_desc(env, agents=None):
    return """
You are an expert in managing distributed energy resources. Controlling distributed energy resources (DERs) entails overseeing a diverse array of small-scale, decentralized energy generation and storage systems to optimize their performance and guarantee a reliable, efficient, and sustainable energy supply. These resources may encompass solar panels, energy storage devices, domestic hot water storage, and other localized energy systems that connect to the main grid or function independently within microgrids or off-grid setups. The control of DERs is a crucial component of contemporary smart grids, which strive to incorporate renewable energy sources, minimize greenhouse gas emissions, and bolster the resilience of the energy infrastructure. 
The primary goal is to devise a controller that can makes decsisions following rules below:
Rule 1. Aim to maintain the consistency of net electricity consumption as much as possible to prevent abrupt fluctuations in sequential electricity usage.
Rule 2. Endeavor to minimize carbon emissions to the greatest extent feasible.
Rule 3: Monitor the indoor temperature and cooling demand continuously. If the indoor temperature exceeds the desired thermal comfort range, activate the cooling device.
Rule 4: Prioritize the use of cooling and domestic hot water (DHW) storage for meeting cooling demands when the stored energy is sufficient. Activate the cooling device only when the cooling and domestic hot water (DHW) storage is unable to meet the demand.
Rule 5: Schedule the cooling and/or domestic hot water (DHW) storage to charge during off-peak hours when electricity demand is low, and electricity from the grid is cheaper. This will help maintain stable electricity consumption from the grid.
Rule 6: Utilize electricity storage to power the cooling device during peak hours when electricity demand is high, and electricity from the grid is more expensive. This will help maintain stable electricity consumption from the grid. \n"""


def get_action_description(env, actions, agents=None):
    selected_agents = agents if agents else env.all_agents
    context = json.dumps({agent: actions[agent] for agent in selected_agents})
    return context


def get_rewards_description(env, rewards, agents=None):
    selected_agents = agents if agents else env.all_agents
    r = np.sum([rewards[agent] for agent in selected_agents])
    return f"{int(10*r)}"
