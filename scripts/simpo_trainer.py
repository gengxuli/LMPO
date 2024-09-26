
from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
import math
from scipy.stats import wasserstein_distance as wd
from scipy.stats import energy_distance as ed
from torchmetrics import HingeLoss
import numpy as np
from trl.trainer.utils import RunningMoments
            

class SimPOTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        self.gamma = training_args.gamma
        self.running = RunningMoments(self.accelerator)

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        length_ratio,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # pi_logratios = policy_chosen_logps - policy_rejected_logps

        # gamma_logratios = self.gamma / self.beta
        # pi_logratios = pi_logratios.to(self.accelerator.device)
        # logits = pi_logratios - gamma_logratios

        # tie_logits = ((r ** 2 - 1) * chosen_probs * rejected_probs) / ((chosen_probs + r * rejected_probs) * (rejected_probs + r * chosen_probs))
        # gamma_logratios = self.gamma / self.beta
        
        """
        new method(考虑主场优势,以原来的B-T模型作为切入点)
        """
        # s = 1
        # h = math.exp(self.beta / s)
        # d = self.beta

        # chosen_probs = torch.exp(policy_chosen_logps / s)
        # rejected_probs = torch.exp(policy_rejected_logps / s)

        """
        with ties
        """
        # chose_chosen_logits = h * chosen_probs / (h * chosen_probs + rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))
        # chose_rejected_logits = rejected_probs / (h * chosen_probs + rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))
        # chose_tie_logits = d * torch.sqrt(h * chosen_probs * rejected_probs) / (h * chosen_probs + rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))

        # reject_chosen_logits = chosen_probs / (chosen_probs + h * rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))
        # reject_rejected_logits = h * rejected_probs / (chosen_probs + h * rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))
        # reject_tie_logits = d * torch.sqrt(h * chosen_probs * rejected_probs) / (chosen_probs + h * rejected_probs + d * torch.sqrt(h * chosen_probs * rejected_probs))

        """
        without ties
        """
        # chose_chosen_logits = h * chosen_probs / (h * chosen_probs + rejected_probs)
        # chose_rejected_logits = rejected_probs / (h * chosen_probs + rejected_probs)

        # reject_chosen_logits = chosen_probs / (chosen_probs + h * rejected_probs)
        # reject_rejected_logits = h * rejected_probs / (chosen_probs + h * rejected_probs)

        # self.k = 1./10.
        # margin = (1 - chosen_probs).mul(1./(1 + torch.exp((chosen_probs - rejected_probs) / self.k)))

        # self.k = 5.
        # margin = (1 - chosen_probs).mul(torch.log((0.5 - chosen_probs / 2 + rejected_probs / 2. + 1e-8) / (0.5 + chosen_probs / 2 - rejected_probs / 2. + 1e-8)) / self.k + 0.5)

        # self.k = 5.
        # margin = (1 - chosen_probs).mul(1 - (chosen_probs - rejected_probs) ** self.k) / 2

        # margin = margin.sum()
        # margin = margin.to(self.accelerator.device)

        # policy_chosen = chose_chosen_logits + reject_chosen_logits
        # policy_rejected = reject_rejected_logits + chose_rejected_logits

        # log_odds_chosen = chose_chosen_logits / (1 - chose_chosen_logits) + reject_chosen_logits / (1 - reject_chosen_logits)
        # log_odds_rejected = chose_rejected_logits / (1 - chose_rejected_logits) + reject_rejected_logits / (1 - reject_rejected_logits)
        # self.alpha = 1
        # self.gamma = 0.02
        # losses = -F.logsigmoid(self.beta * (policy_chosen - policy_rejected)) - F.logsigmoid(-self.alpha * margin)
        # losses = -F.logsigmoid(self.beta * (policy_chosen - policy_rejected)) + self.alpha * margin

        # log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        #     torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        # )
        # lm = policy_chosen_size - policy_rejected_size

        # log_odds = (policy_chosen_logps) / (1 - chosen_probs) - (policy_rejected_logps) / (1 - rejected_probs)
        # ratio = self.beta * F.logsigmoid(log_odds)
        # losses = - ratio + self.alpha * margin

        # chose_logits = reject_chosen_logits  + chose_chosen_logits - chose_tie_logits
        # reject_logits = chose_rejected_logits  + reject_rejected_logits - reject_tie_logits

        # chose_logits = reject_chosen_logits  + chose_chosen_logits 
        # reject_logits = chose_rejected_logits  + reject_rejected_logits

        # chose_logits_cpu = chose_logits.detach().cpu()
        # reject_logits_cpu = reject_logits.detach().cpu()

        # chose_logits_cpu = policy_chosen_logps.detach().cpu()
        # reject_logits_cpu = policy_rejected_logps.detach().cpu()

        # wd1 = wd(chose_logits_cpu, reject_logits_cpu)
        # ed1 = ed(chose_logits_cpu, reject_logits_cpu)
        
        # logits = chose_logits - reject_logits - wd1  
        # logits = logits.to(self.accelerator.device)

        # policy_chosen = torch.exp(policy_chosen_logps)
        # policy_reject = torch.exp(policy_rejected_logps)
        
        # quantic
        # self.k = 5.
        # margin = (1 - chosen_token_logps).mul(torch.abs(1 - (chosen_token_logps - rejected_token_logps) ** self.k)) / 2


        # sigmoid_10
        # self.k = 1./10.

        # self.k = 1.
        # margin = (1 - policy_chosen_logps).mul(1./(1 + torch.exp((policy_chosen_logps - policy_rejected_logps) / self.k)))
        # margin = margin.sum()
        # if active_elements != 0:
        #     margin = margin / active_elements
        # else:
        #     margin = margin
        # margin = margin.to(self.accelerator.device)


        # log_5
        # self.k = 5.
        # margin = (1 - policy_chosen).mul(torch.log((0.5 - policy_chosen / 2 + policy_reject / 2. + 1e-8) / (0.5 + policy_chosen / 2 - policy_reject / 2. + 1e-8)) / self.k + 0.5)

        # policy_chosen_cpu = policy_chosen.detach().cpu()
        # policy_reject_cpu = policy_reject.detach().cpu()
        # wd1 = wd(policy_chosen_cpu, policy_reject_cpu)
        # margin = (1 - policy_chosen) * wd1

        """
        loss_2原来形式
        """
        # logits_chosen = policy_chosen_logps - (policy_chosen_logps + policy_rejected_logps) / 2
        # logits_reject = policy_rejected_logps - (policy_chosen_logps + policy_rejected_logps) / 2


        """
        loss_1 moving average
        """
        # chosen_rewards = self.beta * policy_chosen_logps
        # rejected_rewards = self.beta * policy_rejected_logps
        # rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        # self.running.update(rewards)
        # delta = self.running.mean

        # logits_chosen = logits_chosen.to(self.accelerator.device)
        # logits_reject = logits_reject.to(self.accelerator.device)


        # self.alpha = 1

        """
        loss_2化简形式
        """
        # losses = -self.beta * F.logsigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps) / 2) - F.logsigmoid(-self.alpha * margin)
        # losses = - F.logsigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps)) - F.logsigmoid(-self.alpha * margin)

        # losses =  - F.logsigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps)) - F.logsigmoid(-self.alpha * margin)

        # losses =  - F.logsigmoid(self.beta * policy_chosen_logps) - F.logsigmoid (-self.beta * policy_rejected_logps)

        """
        loss_2原形式
        """
        # losses = -F.logsigmoid(self.beta * logits_chosen) - F.logsigmoid(-self.beta * logits_reject) -F.logsigmoid(-self.alpha * margin)

        """
        loss_1形式
        """
        # losses = -F.logsigmoid((self.beta * policy_chosen_logps) - delta) - F.logsigmoid(-(self.beta * policy_rejected_logps - delta)) -F.logsigmoid(-self.alpha * margin)

        # losses = -F.logsigmoid(self.beta * policy_chosen_logps - delta) - F.logsigmoid(-(self.beta * policy_rejected_logps - delta)) 

        # logits = policy_chosen_logps - policy_rejected_logps - torch.from_numpy(np.array(wd1))
        # logits = policy_chosen_logps - policy_rejected_logps - self.beta * margin
        # logits = - self.beta * margin

        # if self.loss_type == "sigmoid":
        #     losses = (
        #         -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        #         - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        #     )
        # elif self.loss_type == "hinge":
        #     losses = torch.relu(1 - self.beta * logits)
        # elif self.loss_type == "ipo":
        #     # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        #     losses = (logits - 1 / (2 * self.beta)) ** 2
        # else:
        #     raise ValueError(
        #         f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
        #     )
        
        # self.k = 1. / 10.
        # policy_chosen = torch.exp(self.chosen_loss_mask * self.chosen_logps)
        # policy_rejected = torch.exp(self.rejected_loss_mask * self.rejected_logps)

        # if self.chosen_loss_mask.long().sum() > self.rejected_loss_mask.long().sum():
        #     num_active_elements = self.rejected_loss_mask.long().sum()
        # else:
        #     num_active_elements = self.chosen_loss_mask.long().sum()
        # self.gamma = 0.02 

        # margin = - F.logsigmoid(self.gamma * self.length_ratio) * (1 - policy_chosen).mul(1./(1 + torch.exp((policy_chosen - policy_rejected) / self.k))) 

        # margin = margin.sum()
        # margin /= num_active_elements
        # margin = margin.to(self.accelerator.device)

        self.alpha = 1
        self.k = 1. / 10.
        self.gamma = 0.02

        
        policy_chosen = torch.exp(self.chosen_mask.sum(-1) * policy_chosen_logps)
        policy_rejected = torch.exp(self.rejected_mask.sum(-1) * policy_rejected_logps)

        if length_ratio >= 0:
            margin = length_ratio * (1 - policy_chosen).mul(1./(1 + torch.exp((policy_chosen - policy_rejected) / self.k)))
        else:
            margin = length_ratio *  (1. / (1 - policy_chosen)).mul(1 + torch.exp((policy_chosen - policy_rejected) / self.k))
        
        
        losses =  - F.logsigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps) - self.alpha * margin)

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # padding_mask = labels == label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        # padding_mask = padding_mask.view(-1)
        # num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        # num_active_elements = num_active_elements.sum().item() / num_active_elements.numel()

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

        return (per_token_logps * loss_mask).sum(-1), loss_mask


        # return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1), num_active_elements

        # return total_logps, loss_mask.sum(-1)
        # return total_logps, per_token_logps, loss_mask

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss
        
        # if self.is_encoder_decoder:
        #     labels = concatenated_batch["concatenated_labels"].clone()
        # else:
        #     labels = concatenated_batch["concatenated_input_ids"].clone()
        # labels = concatenated_batch["concatenated_labels"].clone()
        # loss_chosen = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])
        # loss_rejected = cross_entropy_loss(all_logits[len_chosen:], labels[len_chosen:])
        # loss_chosen = loss_chosen.mean(dim=list(range(1, len(loss_chosen.shape))))
        # loss_rejected = loss_rejected.mean(dim=list(range(1, len(loss_rejected.shape))))
        # log_odds = (loss_chosen) / (torch.exp(loss_chosen) - 1) - (loss_reject) / (torch.exp(loss_reject) - 1)
        # log_odds_chosen = (loss_chosen) / (torch.exp(loss_chosen) - 1)
        # log_odds_reject = (loss_reject) / (torch.exp(loss_reject) - 1)

        # ratio = torch.relu(1 - log_odds) 
        # ratio_losses = ratio.mean()
        # ratio_losses = loss_chosen
        # ratio_losses = -F.logsigmoid(log_odds_chosen) - F.logsigmoid(-log_odds_reject)
        # ratio_losses = loss_chosen

        # chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        # all_logps = self.get_batch_logps(
        #     all_logits,
        #     concatenated_batch["concatenated_labels"],
        #     average_log_prob=False,
        #     is_encoder_decoder=self.is_encoder_decoder,
        #     label_pad_token_id=self.label_pad_token_id,
        # )

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )


        all_logps = all_logps / size_completion.sum(-1)

        chosen_logps = all_logps[:len_chosen] 
        rejected_logps = all_logps[len_chosen:] 

        chosen_logits = all_logits[:len_chosen] 
        rejected_logits = all_logits[len_chosen:]

        chosen_mask = size_completion[:len_chosen]
        rejected_mask = size_completion[len_chosen:]

        self.chosen_mask = chosen_mask
        self.rejected_mask = rejected_mask

        chosen_length = chosen_mask.sum(-1)
        rejected_length = rejected_mask.sum(-1)

        chosen_length = chosen_length.sum().item() / chosen_length.numel()
        rejected_length = rejected_length.sum().item() / rejected_length.numel()

        # chosen_length = chosen_length.sum() / chosen_length.numel()
        # rejected_length = rejected_length.sum() / rejected_length.numel()

        # chosen_length = chosen_mask.sum(-1)
        # rejected_length = rejected_mask.sum(-1)

        # chosen_length = chosen_mask.sum() / chosen_mask.numel()
        # rejected_length = rejected_mask.sum() / rejected_mask.numel()

        length_dis = chosen_length - rejected_length
        # length_ratio = length_dis

        # if chosen_length >= rejected_length:
        #     self.min_length = rejected_length
        # else:
        #     self.min_length = chosen_length

        if length_dis != 0:
            length_ratio = 1 / length_dis
        else:
            length_ratio = 1

      
        # return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, length_ratio, loss_chosen, loss_rejected)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, length_ratio)

        # return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_size, rejected_size)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        "`""Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            length_ratio,
            # policy_loss_chosen,
            # policy_loss_rejected,
            # policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            length_ratio,
        )

        # self.k = 1. / 10.
        # self.gamma = 0.02 
        # loss = losses.mean() + policy_nll_loss

        # policy_chosen = torch.exp(-policy_loss_chosen)
        # policy_rejected = torch.exp(-policy_loss_rejected)



        # margin = length_ratio * (1 - policy_loss_chosen).mul(1./(1 + torch.exp((policy_loss_chosen - policy_loss_rejected) / self.k)))
        # margin = length_ratio *  policy_loss_chosen.mul(1./(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k)))

        # margin = abs(length_ratio) * policy_loss_chosen.mul(1./(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k)))

        # policy_chosen = torch.exp(self.chosen_loss_mask * self.chosen_logps)
        # policy_rejected = torch.exp(self.rejected_loss_mask * self.rejected_logps)

        # num_active_elements = self.chosen_loss_mask.long().sum()

        # if self.chosen_loss_mask.long().sum() > self.rejected_loss_mask.long().sum():
        #     num_active_elements = self.rejected_loss_mask.long().sum()
        # else:
        #     num_active_elements = self.chosen_loss_mask.long().sum()


        # margin = - F.logsigmoid(self.gamma * length_ratio) * (1 - policy_chosen).mul(1./(1 + torch.exp((policy_chosen - policy_rejected) / self.k))) 

        # margin = length_ratio * (1 - policy_chosen).mul(1./(1 + torch.exp((policy_chosen - policy_rejected) / self.k)))

        # margin = - F.logsigmoid(self.gamma * length_ratio) * policy_loss_chosen.mul(1./(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k)))

        # if length_ratio >= 0:
        #     margin = length_ratio * policy_loss_chosen.mul(1./(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k)))
        # else:
        #     margin = length_ratio *  (1. / policy_loss_chosen).mul(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k))

        # margin = - F.logsigmoid(self.gamma * length_ratio) * policy_loss_chosen.mul(1./(1 + torch.exp((policy_loss_rejected - policy_loss_chosen) / self.k)))

        # if length_ratio >= 0:
        #     margin = length_ratio * (1 - policy_chosen).mul(1./(1 + torch.exp((policy_chosen - policy_rejected) / self.k)))
        # else:
        #     margin = length_ratio * (1. / (1 - policy_chosen)).mul(1 + torch.exp((policy_chosen - policy_rejected) / self.k))

        # self.k = 5.
        # margin = length_ratio * (1 - policy_chosen_logps).mul(1 - (policy_chosen - policy_rejected) ** self.k) / 2
        # margin = margin.sum()

        # if active_elements != 0:
        #     margin = margin / active_elements
        # else:
        #     margin = margin

        # margin = margin.sum()
        # # margin /= self.min_length
        # margin = margin.to(self.accelerator.device)

        # self.alpha = 1
        # loss = (losses - F.logsigmoid(-self.alpha * margin)).mean()
        # loss = (losses + self.alpha * margin).mean()

        loss = losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        # metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()
        return loss, metrics