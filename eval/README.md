## Evaluation
We provide details on the evaluation of the models in this directory. Specifically, we evaluate on AlpacaEval 2 and ArenaHard. AlpacaEval 2 consists of 805 questions from 5 datasets. The most recently released Arena-Hard is an enhanced version of an MT-Bench,
incorporating 500 well-defined technical problem-solving queries. We report scores following each
benchmark’s evaluation protocol. For AlpacaEval 2, we report both the raw win rate (WR) and the
length-controlled win rate (LC). The LC metric is specifically designed to be robust against model verbosity. For Arena-Hard, we report the win rate (WR) against the baseline model. 

### AlpacaEval 2
We provide generation configurations for the released models in the `alpacaeval2/configs` directory, and the corresponding generation templates can be found in `alpacaeval2/templates`. To evaluate the models on AlpacaEval 2, please use the [`alpaca-eval`](https://github.com/tatsu-lab/alpaca_eval) package.

### Arena-Hard
We provide generation configurations for the released models in the `arenahard/configs` directory, and the corresponding generation templates can be found in `arenahard/templates`. To evaluate the models on Arena-Hard, please use the [`arena-hard-auto`](https://github.com/lm-sys/arena-hard-auto) package.

