# Length-Controlled Margin-Based Preference Optimization (LMPO)
Official implementation for the paper titled ["Length-Controlled Margin-Based Preference Optimization without Reference Model"](https://arxiv.org/abs/2502.14643).

<img src="./lmpo.png" width="1000px"></img>

## Ⅰ.Install Requirements

Our codebase is built upon the [alignment-handbook](https://github.com/huggingface/alignment-handbook) repo. The following steps will guide you through the installation process.

First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.2.2`. Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies of [alignment-handbook](https://github.com/huggingface/alignment-handbook) as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

## Ⅱ.Training Scripts

We provide four training config files for the four training setups reported in our paper. The training config is set for 4xA100 GPUs. You may need to adjust `num_processes` and `per_device_train_batch_size` based on your computation environment. 


* Mistral-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_lmpo.py training_configs/mistral-7b-base-lmpo.yaml
```
* Mistral-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_lmpo.py training_configs/mistral-7b-instruct-lmpo.yaml
```
* Llama3-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_lmpo.py training_configs/llama-3-8b-base-lmpo.yaml
```
* Llama3-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_lmpo.py training_configs/llama-3-8b-instruct-lmpo.yaml
```

## Ⅲ.Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, Open LLM Leadboard (v1) and Open LLM Leadboard (v2) as follows (more details can be found under [the eval directory](https://github.com/gengxuli/LMPO/tree/main/eval)):

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

*  Open LLM Leadboard (v1): Please refer to to the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) and [Open LLM Leadboard v1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) for evaluation.

*  Open LLM Leadboard (v2): Please refer to to the [Language Model Evaluation Harness](https://github.com/huggingface/lm-evaluation-harness/tree/adding_all_changess) and [Open LLM Leadboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)  for evaluation.

## Ⅳ.Acknowledgement
The project is built upon

* [SimPO](https://github.com/princeton-nlp/SimPO).

* [The Alignment Handbook](https://github.com/huggingface/alignment-handbook)

* [TRL-Transformer Reinforcement Learning](https://github.com/huggingface/trl).

## Ⅴ.Citation
If you find this code useful for your research, please cite our papers
```
@article{li2025length,
  title={Length-Controlled Margin-Based Preference Optimization without Reference Model},
  author={Li, Gengxu and Xia, Tingyu and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2502.14643},
  year={2025}
}
```
