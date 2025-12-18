from collections import OrderedDict
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments.overcooked.overcooked import Overcooked
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict
import pdb

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
from jaxmarl.environments.overcooked.layouts import single_cramped_room


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 10, # reward for putting ingredients 
    "PLATE_PICKUP_REWARD": 3, # reward for picking up a plate
    "SOUP_PICKUP_REWARD": 15, # reward for picking up a ready soup
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
    done = 6


@struct.dataclass
class State:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23 # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20 # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3 # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40 # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class BlindOvercooked(Overcooked):
    """Blind Overcooked -- one agent is blind and can only see the items it runs into"""
    def __init__(
            self, **kwargs
    ):
        super().__init__(**kwargs)

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array([actions["agent_0"], actions["agent_1"]]))

        state, reward, shaped_reward = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {"agent_0": reward, "agent_1": reward}
        shaped_rewards = {"agent_0": shaped_reward[0], "agent_1": shaped_reward[1]}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards},
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation, of size (height x width x n_layers), where n_layers = 26.
        Layers are of shape (height x width) and  are binary (0/1) except where indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions (1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either a cooking/done pot or to a soup (dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2] ### TESTING ADDING CHANNEL INDICATING WHICH AGENT IS WHICH
        padding = (state.maze_map.shape[0]-height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)    # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc   # 0/3, as long as cooking or done
        pot_cooking_time_layer = pot_status * (pot_status < POT_FULL_STATUS)                           # Timer: 19 to 0
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc                 # Ready soups, plated or not
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0, state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1, state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(state.agent_inv,(1,2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers,0), agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers,0)
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers,0)

        env_layers = [
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),       # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),        # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),                     # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes
            urgency_layer,                                                          # 25
        ]

        # Agent related layers
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layer_idx = state.agent_dir_idx+jnp.array([0,4])
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx,:,:].set(agent_pos_layers)

        blind_mask = jnp.zeros((height, width), dtype=jnp.uint8)
        blind_mask = blind_mask.at[state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)

        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(agent_pos_layers[1])

        
        bob_obs = bob_obs.at[1].set(jnp.zeros_like(agent_pos_layers[0]))
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:])
        bob_obs = bob_obs.at[6:10].set(jnp.zeros_like(agent_direction_layers[0:4]))
        
        blind_env_layers = [layer * blind_mask for layer in env_layers]
        bob_obs = bob_obs.at[10:].set(jnp.stack(blind_env_layers))


        alice_obs = jnp.transpose(alice_obs, (1, 2, 0))
        bob_obs = jnp.transpose(bob_obs, (1, 2, 0))

        return {"agent_0" : alice_obs, "agent_1" : bob_obs}

    def step_agents(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, float]:

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal : jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],                     # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],                     # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[1],
            bob_pos
        )

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward, alice_shaped_reward = self.process_interact(maze_map, state.wall_map, fwd_pos, state.agent_inv, 0)
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact,
                              candidate_maze_map,
                              maze_map)
        alice_inv = jax.lax.select(alice_interact,
                              alice_inv,
                              state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.)
        alice_shaped_reward = jax.lax.select(alice_interact, alice_shaped_reward, 0.)

        candidate_maze_map, bob_inv, bob_reward, bob_shaped_reward = self.process_interact(maze_map, state.wall_map, fwd_pos, state.agent_inv, 1)
        maze_map = jax.lax.select(bob_interact,
                              candidate_maze_map,
                              maze_map)
        bob_inv = jax.lax.select(bob_interact,
                              bob_inv,
                              state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.)
        bob_shaped_reward = jax.lax.select(bob_interact, bob_shaped_reward, 0.)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status-1) + (~is_cooking) * pot_status # defaults to zero if done
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        reward = alice_reward + bob_reward

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            reward,
            (alice_shaped_reward, bob_shaped_reward)
        )


        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "BlindOvercooked"


if __name__ == "__main__":
    env = BlindOvercooked( 
            layout = None,
            random_reset= True,
            max_steps= 256,
            single_agent= False,
            check_held_out= False,
            shuffle_inv_and_pot= True)

    from jaxmarl.viz.overcooked_jitted_visualizer import render_fn
    import imageio


    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    def render_reset(key):
        obs, state = env.reset(key)
        return render_fn(state)
    images = jax.vmap(render_reset)(keys)
    # for each image, save it as a png
    for i, image in enumerate(images):
        imageio.imwrite(f"image_{i}.png", image)
        print(f"Saved image_{i}.png")
