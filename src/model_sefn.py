from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import Config as cfg


class SEFNModule(nn.Module):
    """Module III - Self-Evolving Financial Narrative.
    """

    def __init__(self):
        super().__init__()
        print(f"[SEFN] Loading LLM: {cfg.llm_model}")
        self.tokenizer_llm = AutoTokenizer.from_pretrained(cfg.llm_model)
        if self.tokenizer_llm.pad_token is None:
            self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model,
            torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        self.sep_adapter = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.llm_hidden),
            nn.Tanh(),
            nn.Linear(cfg.llm_hidden, cfg.llm_hidden),
        )
        self.value_head = nn.Linear(cfg.llm_hidden, 1)
        self.error_memory: List[Dict] = []

    def encode_prefix(self, quant_factors: torch.Tensor) -> torch.Tensor:
        return self.sep_adapter(quant_factors.float()).unsqueeze(1)

    def _label_name(self, label: int) -> str:
        if 0 <= label < len(cfg.label_names):
            return cfg.label_names[label]
        return f"CLASS_{label}"

    @torch.no_grad()
    def generate_explanation(
        self,
        quant_factors: torch.Tensor,
        pred_label: torch.Tensor,
        true_label: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        log_vol: Optional[torch.Tensor] = None,
        var_est: Optional[torch.Tensor] = None,
        max_new: int = cfg.max_explain_len,
    ) -> List[str]:
        B = quant_factors.size(0)
        prefix_emb = self.encode_prefix(quant_factors)
        explanations: List[str] = []

        for i in range(B):
            pred_name = self._label_name(int(pred_label[i].item()))
            conf_txt = "" if confidence is None else f", confidence={float(confidence[i].item()):.2%}"
            risk_txt = ""
            if log_vol is not None and var_est is not None:
                risk_txt = f", log_vol={float(log_vol[i].item()):.4f}, VaR={float(var_est[i].item()):.4f}"

            prompt = (
                "Generate a concise stock analysis report. "
                f"Prediction={pred_name}{conf_txt}{risk_txt}. "
                "Explain using news signal, market status, price trend, and risk. Report:"
            )
            enc = self.tokenizer_llm(prompt, return_tensors="pt", truncation=True, max_length=96)
            enc = enc.to(quant_factors.device)
            token_embeds = self.llm.get_input_embeddings()(enc.input_ids)
            combined = torch.cat([prefix_emb[i : i + 1].to(token_embeds.dtype), token_embeds], dim=1)
            attention_mask = torch.ones((1, combined.size(1)), dtype=torch.long, device=combined.device)

            out = self.llm.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tokenizer_llm.pad_token_id,
            )
            text = self.tokenizer_llm.decode(out[0], skip_special_tokens=True)

            if true_label is not None and pred_label[i] != true_label[i]:
                true_name = self._label_name(int(true_label[i].item()))
                self.error_memory.append(
                    {
                        "factors": quant_factors[i].detach().cpu(),
                        "pred": int(pred_label[i].item()),
                        "true": int(true_label[i].item()),
                    }
                )
                text += f" [POST-HOC CHECK: actual label was {true_name}.]"

            explanations.append(text)
        return explanations

    def compute_policy_logprobs(
        self,
        quant_factors: torch.Tensor,
        explanation_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Used only when real explanation_ids are available for PPO/RLHF."""
        prefix_emb = self.encode_prefix(quant_factors)
        token_embs = self.llm.get_input_embeddings()(explanation_ids.long())
        prefix_emb = prefix_emb.to(token_embs.dtype)
        combined = torch.cat([prefix_emb, token_embs], dim=1)
        outputs = self.llm(inputs_embeds=combined, output_hidden_states=True)
        logits = outputs.logits[:, :-1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, explanation_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        hidden = outputs.hidden_states[-1][:, 0, :].float()
        values = self.value_head(hidden).squeeze(-1)
        return token_log_probs, values


def compute_reward(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    quant_factors: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None,
    explanation_factors: Optional[torch.Tensor] = None,
    alpha: float = 0.8,
    beta: float = 0.2,
) -> torch.Tensor:
    """Reward helper for future PPO experiments.

    Default is classification reward only. Factor alignment is added only when a
    real explanation encoder supplies explanation_factors.
    """
    correct = (pred_labels == true_labels).float()
    acc_reward = correct * 2.0 - 1.0

    if quant_factors is not None and explanation_factors is not None:
        align_reward = F.cosine_similarity(
            F.normalize(quant_factors.float(), dim=-1),
            F.normalize(explanation_factors.float(), dim=-1),
            dim=-1,
        )
    else:
        align_reward = torch.zeros_like(acc_reward)

    return alpha * acc_reward + beta * align_reward


class PPOTrainer:
    """PPO trainer reserved for real explanation tokens + real reward model."""

    def __init__(self, sefn: SEFNModule, lr: float = 1e-5):
        params = [p for p in sefn.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def update(
        self,
        sefn: SEFNModule,
        quant_factors: torch.Tensor,
        explanation_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, float]:
        rewards = torch.clamp(rewards.float(), -5.0, 5.0)
        values = values.float()
        old_log_probs = old_log_probs.float()
        advantages = rewards - values.detach()
        adv_std = advantages.std().clamp_min(1e-6)
        advantages = ((advantages - advantages.mean()) / adv_std).clamp(-5.0, 5.0)

        total = 0.0
        updates = 0
        last_policy = 0.0
        last_value = 0.0
        for _ in range(cfg.ppo_epochs):
            new_log_probs, new_values = sefn.compute_policy_logprobs(quant_factors, explanation_ids)
            old_mean = old_log_probs.mean(dim=-1)
            new_mean = new_log_probs.mean(dim=-1)
            ratio = torch.exp((new_mean - old_mean.detach()).clamp(-10.0, 10.0))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            returns = rewards + cfg.gamma * new_values.detach()
            value_loss = F.smooth_l1_loss(new_values, returns)
            loss = policy_loss + 0.5 * value_loss
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in sefn.parameters() if p.requires_grad], cfg.grad_clip)
            self.optimizer.step()
            total += float(loss.item())
            last_policy = float(policy_loss.item())
            last_value = float(value_loss.item())
            updates += 1

        if updates == 0:
            return {"ppo_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        return {"ppo_loss": total / updates, "policy_loss": last_policy, "value_loss": last_value}
