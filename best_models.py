"""
Configuration file for the best performing models.
This file allows run.py to automatically load the best model for a given environment and algorithm.
"""

BEST_MODELS = {
    # ALE/Breakout-v5
    "ALE/Breakout-v5": {
        "dqn": {
            "dqn": {
                "model_path": "results/Atari/ALE/Breakout-v5/DQN_Vanilla/lr0.0001_uf1000_sd42_hd256_bs32_env16_20260113-033247/models/final_model.pth",
                "hidden_dim_dqn": 256,
            },
            "double": {
                "model_path": "results/Atari/ALE/Breakout-v5/DQN_Double/lr5e-05_uf1000_sd42_hd512_bs32_env16_20260114-045355/models/final_model.pth",
                "hidden_dim_dqn": 512,
            },
            "dueling": {
                "model_path": "results/Atari/ALE/Breakout-v5/DQN_Dueling/lr0.0001_uf5000_sd42_hd512_bs32_env16_20260113-154856/models/final_model.pth",
                "hidden_dim_dqn": 512,
            },
            "rainbow": {
                "model_path": "results/Atari/ALE/Breakout-v5/DQN_Rainbow/lr0.0001_uf5000_sd42_hd512_bs32_env16_20260114-165851/models/model_4800000.pth",
                "hidden_dim_dqn": 512,
            },
        }
    },

    "ALE/Pong-v5": {
        "dqn": {
            "dqn": {
                "model_path": "results/Atari/ALE/Pong-v5/DQN_Vanilla/lr5e-05_uf1000_sd42_hd512_bs32_env16_20260115-100721/models/final_model.pth",
                "hidden_dim_dqn": 512,
            },
            "double": {
                "model_path": "results/Atari/ALE/Pong-v5/DQN_Double/lr0.0001_uf2000_sd42_hd512_bs32_env16_20260115-125634/models/final_model.pth",
                "hidden_dim_dqn": 512,
            },
            "dueling": {
                "model_path": "results/Atari/ALE/Pong-v5/DQN_Dueling/lr5e-05_uf1000_sd42_hd256_bs32_env16_20260115-152906/models/final_model.pth",
                "hidden_dim_dqn": 256,
            },
            "rainbow": {
                "model_path": "results/Atari/ALE/Pong-v5/DQN_Rainbow/lr0.0001_uf1000_sd42_hd256_bs32_env16_20260115-070355/models/final_model.pth",
                "hidden_dim_dqn": 256,
            },
        }
    },

    # HalfCheetah-v4
    "HalfCheetah-v4": {
        "ppo": {
            "model_path": "results/MuJoCo/HalfCheetah-v4/PPO_Standard/lra5e-05_lrc0.0002_clp0.2_sd42_hd256_env16_20260112-054627/models/final_model.pth", # Placeholder
            "obs_rms_path": None, # Placeholder
            "hidden_dim_ppo": 256,
        }
    },

    # Hopper-v4
    "Hopper-v4": {
        "ppo": {
            "model_path": "results/MuJoCo/Hopper-v4/PPO_Standard/lra5e-05_lrc0.0001_clp0.1_sd101_hd256_env16_20260117-091620/models/model_1850000.pth", # Placeholder
            "obs_rms_path": "results/MuJoCo/Hopper-v4/PPO_Standard/lra5e-05_lrc0.0001_clp0.1_sd101_hd256_env16_20260117-091620/models/obs_rms_1850000.pkl", # Placeholder
            "hidden_dim_ppo": 256,
        }
    },
    
    # Ant-v4
    "Ant-v4": {
        "ppo": {
            "model_path": "results/MuJoCo/Ant-v4/PPO_Standard/lra5e-05_lrc0.0002_clp0.2_sd101_hd256_env16_20260117-091620/models/final_model.pth", # Placeholder
            "obs_rms_path": "results/MuJoCo/Ant-v4/PPO_Standard/lra5e-05_lrc0.0002_clp0.2_sd101_hd256_env16_20260117-091620/models/final_obs_rms.pkl", # Placeholder
            "hidden_dim_ppo": 256,
        }
    }
}