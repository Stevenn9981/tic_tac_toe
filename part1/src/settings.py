num_iterations = 2000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
initial_collect_episodes = 5  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
collect_episodes_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

batch_size = 128  # @param {type:"integer"}
learning_rate = 5e-4  # @param {type:"number"}
log_interval = 5  # @param {type:"integer"}

num_eval_episodes = 50  # @param {type:"integer"}
eval_interval = 20  # @param {type:"integer"}

gamma = 0  # @param {type:"number"}
n_step_update = 1  # @param {type:"integer"}
fc_layer_params = (100,)

BOARD_SIZE = 9  # @param {type:"integer"}

REWARD_DRAW = 0  # @param {type:"number"}
REWARD_ALIVE = 0  # @param {type:"number"}
REWARD_NON_ADJ = -0.03  # @param {type:"number"}

# '_' means empty position, 'O' and 'X' means two players.
REWARD_ACTIVE_TWO = 0.1  # @param {type:"number"} _OO_ or _XX_, we call it active_two
REWARD_NONACT_THREE = 0.3  # @param {type:"number"} _OOOX or _XXXO or XOOO_ or OXXX_, we call it non_active_three
REWARD_ACTIVE_THREE = 0.9  # @param {type:"number"} _OOO_ or _XXX_, we call it active_three
REWARD_WIN = 3  # @param {type:"number"} _OOOO_ or _XXXX_

tempdir = "./"