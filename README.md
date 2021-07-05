## Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games (CoG 2021)

This repo contains the code for the paper [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807).

[Full paper](https://arxiv.org/abs/2105.13807) | [Blog Post](https://wandb.ai/vwxyzjn/gym-microrts-paper/reports/Gym-RTS-Toward-Affordable-Deep-Reinforcement-Learning-Research-in-Real-Time-Strategy-Games--Vmlldzo2MDIzMTg)

## Get started

Make sure you have `ffmpeg` and `jdk>=1.8.0` installed. Then install the dependencies:

```bash
git clone https://github.com/vwxyzjn/gym-microrts-paper
cd gym-microrts-paper
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce and plot results


### Depreciation note

Note that the experiments are done with [`gym_microrts==0.3.2`](https://github.com/vwxyzjn/gym-microrts/tree/v0.3.2). As we move forward beyond `v0.4.x`, we are planing to deprecate UAS despite its better performance in the paper. This is because UAS has more complex implementation and makes it really difficult to incorporate selfplay or imitation learning in the future.


### UAS experiments:
PPO + invalid action masking + diverse bots + IMPALA-CNN (our best agent)
```bash
python ppo_diverse_impala.py --capture-video
```

PPO + invalid action masking  + diverse bots
```bash
python ppo_diverse.py --capture-video
```

PPO + invalid action masking
```bash
python ppo_coacai.py --capture-video
```

PPO + partial invalid action masking
```bash
python ppo_coacai_partial_mask.py --capture-video
```

PPO
```bash
python ppo_coacai_no_mask.py --capture-video
```



### Gridnet experiments:
PPO + invalid action masking +half self-play / half bots + encoder-decoder
```bash
python ppo_gridnet_diverse_encode_decode.py --capture-video  --num-bot-envs 8 --num-selfplay-envs 16  --exp-name ppo_gridnet_selfplay_diverse_encode_decode
```

PPO + invalid action masking + selfplay + encoder-decoder
```bash
python ppo_gridnet_diverse_encode_decode.py --capture-video  --num-bot-envs 0 --num-selfplay-envs 24  --exp-name ppo_gridnet_selfplay_encode_decode
```

PPO + invalid action masking + diverse bots + encoder-decoder
```bash
python ppo_gridnet_diverse_encode_decode.py --capture-video
```

PPO + invalid action masking  + diverse bots + IMPALA-CNN
```bash
python ppo_gridnet_diverse_impala.py --capture-video
```

PPO + invalid action masking  + diverse bots
```bash
python ppo_gridnet_diverse.py --capture-video
```

PPO + invalid action masking
```bash
python ppo_gridnet_coacai.py --capture-video
```

PPO + partial invalid action masking
```bash
python ppo_gridnet_coacai_partial_mask.py --capture-video
```

PPO
```bash
python ppo_gridnet_coacai_no_mask.py --capture-video
```

### Experiment management

We use [Weights and Biases](https://wandb.com) for experiments management, which 
syncs the training metrics, videos of the agents playing the game, and trained models
of our script.

You can enable this feature by toggling the `--prod-mode` tag with the scripts above. 
For example, try running

```
python ppo_diverse_impala.py --capture-video --prod-mode --wandb-project gym-microrts-paper
```

and you should see ouputs similar to the following

```bash
wandb: Currently logged in as: costa-huang (use `wandb login --relogin` to force relogin)
wandb: wandb version 0.10.25 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.10.24
wandb: Syncing run MicrortsDefeatCoacAIShaped-v3__ppo_diverse_impala__1__1618184644
wandb: ⭐️ View project at https://wandb.ai/vwxyzjn/gym-microrts-paper
wandb: 🚀 View run at https://wandb.ai/vwxyzjn/gym-microrts-paper/runs/2gw2f8tl
wandb: Run data is saved locally in /home/costa/Documents/work/go/src/github.com/vwxyzjn/gym-microrts-paper/wandb/run-20210411_194404-lokq7jxs
wandb: Run `wandb offline` to turn off syncing.
```

### Evaluations

Once the agents are trained with `--prod-mode` toggled on, you can go to the experiment page to download the trained model, which we can use for evaluation. For example, you can download [this experiment](https://wandb.ai/vwxyzjn/gym-microrts-paper/runs/2gw2f8tl/files?workspace=user-costa-huang)'s `agent.pt`.

This repo comes with pre-trained models at the `trained_models` directory. To run evaluation for `PPO + invalid action masking + diverse bots + IMPALA-CNN`, for example, try running

```bash
curl -O https://microrts.s3.amazonaws.com/microrts/gym-microrts-paper/trained_models.zip &&unzip trained_models.zip
python agent_eval.py --exp-name ppo_diverse_impala \
    --agent-model-path trained_models/ppo_diverse_impala/agent-2.pt \
    --max-steps 4000 --num-eval-runs 100 \
    --wandb-project-name gym-microrts-paper-eval \
    --prod-mode --capture-video
```

To see how we run all the evaluations, check out `agent_eval.sh`.

### Plots

Check out the code in the `plots` folder. Try running

```
curl -O https://microrts.s3.amazonaws.com/microrts/gym-microrts-paper/all_data.csv && mv all_data.csv plots/all_data.csv
python plot_ablation.py
python plot_all.py
python plot_hist.py
python plot_shaped_vs_sparse.py
python plot_uas_vs_gridnet.py
```

The CSV data is obtained either through the [`wandb` export APIs](https://docs.wandb.ai/library/public-api-guide) or directly at the `wandb` dashboard such as the ["Ablation Studies" report](https://wandb.ai/vwxyzjn/gym-microrts-paper-eval/reports/Ablation-Studies--Vmlldzo1MjU2MjE)

