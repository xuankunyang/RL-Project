"""
Configuration file for the best performing models.
This file allows run.py to automatically load the best model for a given environment and algorithm.
"""

BEST_MODELS = {
    # ALE/Breakout-v5
    "ALE/Breakout-v5": {
        "dqn": {
            "dqn": {
                "model_path": "checkpoints/Breakout_best_dqn.pth",
                "hidden_dim_dqn": 512,
            },
            "double": {
                "model_path": "checkpoints/Breakout_best_double.pth",
                "hidden_dim_dqn": 512,
            },
            "dueling": {
                "model_path": "checkpoints/Breakout_best_dueling.pth",
                "hidden_dim_dqn": 512,
            },
            "rainbow": {
                "model_path": "checkpoints/Breakout_best_rainbow.pth",
                "hidden_dim_dqn": 512,
            },
        }
    },
    
    # Hopper-v4
    "Hopper-v4": {
        "ppo": {
            "model_path": "checkpoints/Hopper_best_ppo.pth", # Placeholder
            "obs_rms_path": "checkpoints/Hopper_best_obs_rms.pkl", # Placeholder
            "hidden_dim_ppo": 256,
        }
    },
    
    # Ant-v4
    "Ant-v4": {
        "ppo": {
            "model_path": "checkpoints/Ant_best_ppo.pth", # Placeholder
            "obs_rms_path": "checkpoints/Ant_best_obs_rms.pkl", # Placeholder
            "hidden_dim_ppo": 256,
        }
    }
}