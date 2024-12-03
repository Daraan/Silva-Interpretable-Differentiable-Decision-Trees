import argparse
import ray
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import (
    PPOTorchRLModule
)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
import gymnasium as gym
import torch
from tqdm import trange

from interpretable_ddts.agents.ddt import DDTCatalog
from interpretable_ddts.agents.ddt_ppo_module import DDTModule
from packaging.version import parse as parse_version, Version

RAY_VERSION = parse_version(ray.__version__)

if __name__ == "__main__":
    # full parser see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='ddt')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
    parser.add_argument("-gpu", "--gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-r", "--rule_list", help="Use rule list setup", action='store_true', default=False)
    parser.add_argument("-s", "--seed", help="Seed", default=-1, type=int)
    parser.add_argument("-np", "--not_parallel", help="Do not run in parallel", action='store_true', default=False)
    parser.add_argument("-p", "--process_number", help="Process number", type=int, default=0)
    parser.add_argument("--silent", help="supress prints", action="store_true", default=False,)
    parser.add_argument("--test", "--dry-run", help="Do not save any models", action="store_true", default=False)
    parser.add_argument(
        "-rl",
        "--rllib",
        help="Use rllib",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    SEED = args.seed
    AGENT_TYPE: str = args.agent_type  # 'ddt', 'mlp'
    NUM_EPS: int = args.episodes  # num episodes Default 1000
    ENV_TYPE: str = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false

    if ENV_TYPE == 'lunar':
        init_env = gym.make('LunarLander-v2')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
        env = "LunarLander-v2"
    elif ENV_TYPE == 'cart':
        init_env = gym.make('CartPole-v1')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
        env = "CartPole-v1"
    else:
        raise Exception('No valid environment selected')

    # Create a single agent RL module spec.
    module_spec = RLModuleSpec(
        module_class=DDTModule,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config={
            #"custom_model": RLlibDDT,
            "custom_model_config": {
                "ddt_agent_config" : {
                    "bot_name": AGENT_TYPE + ENV_TYPE,
                    "input_dim": dim_in,
                    "output_dim": dim_out,
                    "rule_list": args.rule_list,
                    "num_rules": args.num_leaves,
                    "save_output": not args.test,
                    "use_gpu": USE_GPU,
                    "vf_double_output": True,
                },
            },
        },
        catalog_class=DDTCatalog,
    )
    #module = module_spec.build()

    config = PPOConfig()
    config.environment(env)    
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.resources(
        #num_gpus=1 if USE_GPU else 0,
        num_cpus_for_main_process=1,
        #num_learner_workers=0 if args.not_parallel else 4,
        #num_cpus_per_learner_worker=1,
        #num_cpus_per_worker=1,
    )
    if hasattr(config, "env_runners"):
        config.env_runners(
            num_env_runners=0 if args.not_parallel else 8,
            num_cpus_per_env_runner=1,  # num_cpus_per_worker
            )
        config.learners(
            # for fractional GPUs, you should always set num_learners to 0 or 1
            num_learners=0 if args.not_parallel else 4,
            num_cpus_per_learner=1,
            num_gpus_per_learner=1 if USE_GPU else 0,
        )
    else:
        config.rollouts(num_rollout_workers=0 if args.not_parallel else 8)
    config.framework("torch").training(
        learner_class=SilvaLearner,  #
        learner_config_dict={
            "use_silva_loss" : True
        },
        gamma=0.99,
        use_critic=True,
        # with a growing number of Learners and to increase the learning rate as follows:
        # lr = [original_lr] * ([num_learners] ** 0.5)
        lr=1e-3,
        # Sheduled LR
        # lr=[
        #    [0, 1e-5],  # <- initial value at timestep 0
        #    [1000000, 1e-4],  # <- final value at 1M timesteps
        # ],
        clip_param=0.2,
        grad_clip=0.5,
        # grad_clip_by="norm",
        entropy_coeff=0.01,
        # train_batch_size=32, old API
        train_batch_size_per_learner=36,
        # The total effective batch size is then
        # `num_learners` x `train_batch_size_per_learner` and you can
        # access it with the property `AlgorithmConfig.total_train_batch_size`.
        minibatch_size=8,
        num_epochs=20,
        use_kl_loss=False,
    ).rl_module(
        rl_module_spec=module_spec,
    )
    if hasattr("config", "env_runners") or RAY_VERSION >= Version("2.20"):
        config.evaluation(evaluation_num_env_runners=0)  # type: ignore
    else:
        config.evaluation(evaluation_num_workers=0)
    algo = config.build()
    config.validate_train_batch_size_vs_rollout_fragment_length()
    
    # Start dashboard
    if False:
        context = ray.init()
        print(context.dashboard_url)
    
    pbar = trange(args.episodes)
    for i in pbar:
        result = algo.train()
        pbar.set_description(
            f"Mean reward: {result['env_runners']['agent_episode_returns_mean']['default_agent']:.2f} |"
            f"Max reward: {result['env_runners']['episode_return_max']:.0f} |"
            f"Length Avg: {result['env_runners']['episode_len_mean']:.1f} |"
            #f"Loss: {result['learners']['default_policy']['total_loss']:.2f}"
        )
    breakpoint()

    # result.keys()
    # dict_keys(['timers', 'env_runners', 
    # 'num_agent_steps_sampled_lifetime', 'num_env_steps_sampled_lifetime', 'num_episodes_lifetime', 
    # 'learners', 'num_env_steps_trained_lifetime', 'fault_tolerance', 'done', 'training_iteration', 
    # 'trial_id', 'date', 'timestamp', 
    # 'time_this_iter_s', 'time_total_s', 
    # 'pid', 'hostname', 'node_ip', 'config', 'time_since_restore', 'iterations_since_restore', 'perf'])
