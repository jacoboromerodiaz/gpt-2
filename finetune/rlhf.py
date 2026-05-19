import os
import time

import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gpt2.train import get_lr, load_checkpoint, setup_device
from gpt2.utils import unwrap_model
from finetune.sft import extend_encoder
from finetune.data import AlpacaPromptDataset, collate_prompts, IM_END, EOS

RM_NAME = "OpenAssistant/reward-model-deberta-v3-base"


def build_action_mask(sequence_ids, prompt_len, stop_tokens=(IM_END, EOS)):
    """
    Mark completion tokens up to and INCLUDING the first stop token per row.
    """
    B, T = sequence_ids.shape
    mask = torch.zeros(B, T - 1, dtype=torch.bool, device=sequence_ids.device)
    completions = sequence_ids[:, prompt_len:].tolist()
    for i, comp in enumerate(completions):
        stop_pos = next((j for j, t in enumerate(comp) if t in stop_tokens), len(comp))
        n_keep = min(stop_pos + 1, len(comp))  # include the stop token
        mask[i, prompt_len - 1 : prompt_len - 1 + n_keep] = True
    return mask


def compute_sequence_log_probs(model, sequence_ids, action_mask, device_type):
    """
    sequence_ids: (B, T, ) prompt + completion
    action_mask: (B, T-1) True for completion tokens
    Returns: (B, T-1) per-token log probs zeroed outside the mask
    """
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(sequence_ids[:, :-1])  # (B, T-1, vocab_size)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)  # (B, T-1, 1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(
        -1
    )  # (B, T-1)
    return token_log_probs * action_mask.float()


# pylint: disable=too-many-locals
@torch.no_grad()
def compute_rewards(sequence_ids, prompt_len, enc_extended, rm_model, rm_tokenizer):
    """
    sequence_ids: (B*G, T) prompt + completion
    prompt_lens: (B*G,) prompt tokens per row
    Returns: (B*G,) tensor of scalar rewards
    """
    prompt_texts, completion_texts = [], []

    for ids in sequence_ids:
        ids_list = ids.tolist()
        prompt_ids = ids_list[:prompt_len]
        completion_ids = ids_list[prompt_len:]

        stop_pos = next(
            (j for j, t in enumerate(completion_ids) if t in (IM_END, EOS)),
            len(completion_ids),
        )
        completion_ids = completion_ids[:stop_pos]

        prompt_text = enc_extended.decode(prompt_ids)
        prompt_text = (
            prompt_text.replace("<|im_start|>user\n", "")
            .replace("<|im_end|>", "")
            .replace("<|im_start|>assistant\n", "")
            .replace("<|endoftext|>", "")  # left-padding tokens
            .strip()
        )
        completion_text = enc_extended.decode(completion_ids).strip()

        prompt_texts.append(prompt_text)
        completion_texts.append(
            completion_text if completion_text else " "
        )  # avoid empty

    inputs = rm_tokenizer(
        prompt_texts,
        completion_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(rm_model.device)

    with torch.autocast(device_type=rm_model.device.type, dtype=torch.bfloat16):
        scores = rm_model(**inputs).logits.squeeze(-1)  # (B*G,)

    return scores.float()


def grpo_advantages(rewards, group_size, eps=1e-4):
    """
    rewards : (B, G)
    Returns : (B * G,)
    """
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def policy_gradient_loss(
    log_probs_new,
    log_probs_old,
    advantages,
    action_mask,
    clip_eps=0.2,
    fixed_len_norm=False,
    L_max=128,
):
    adv = advantages.unsqueeze(-1)  # broadcast over sequence length
    ratio = torch.exp(log_probs_new - log_probs_old.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * adv, clipped * adv)
    mask = action_mask.float()
    denom = L_max if fixed_len_norm else mask.sum(dim=1).clamp(min=1.0)
    loss = -((surrogate * mask).sum(dim=1) / denom).mean()
    n_tokens = mask.sum().clamp(min=1)
    clip_frac = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
    clip_frac = (clip_frac * mask).sum() / n_tokens
    return loss, clip_frac.item()


def kl_penalty_k3(
    log_probs_new, log_probs_ref, action_mask, fixed_len_norm=False, L_max=128
):
    log_ratio = log_probs_ref - log_probs_new
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    mask = action_mask.float()
    denom = L_max if fixed_len_norm else mask.sum(dim=1).clamp(min=1.0)
    return ((per_token_kl * mask).sum(dim=1) / denom).mean()


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    enc_extended = extend_encoder(enc)

    ctx = setup_device()
    ddp = ctx.ddp

    max_lr = 3e-6
    min_lr = 2e-7
    warmup_steps = 300
    weight_decay = 0.1
    clip_eps = 0.1
    beta_kl = 0.4  # KL coefficient
    inner_update_steps = 2  # PPO-style updates per rollout
    group_size = 16  # completions per prompt
    max_new_tokens = 256
    temperature = 0.8
    top_k = 50
    batch_size = 1  # prompts per micro-step
    sim_batch_size = 64
    assert sim_batch_size % (batch_size * group_size) == 0
    grad_accum_steps = sim_batch_size // (batch_size * group_size)
    epochs = 1
    fixed_len_norm = True  # divide by L_max instead of per-sequence length

    checkpoint_file = os.environ.get(
        "GRPO_CHECKPOINT",
        "/workspace/gpt-2/finetune/log/sft_model_05615.pt",
    )

    model, _ = load_checkpoint(checkpoint_file, ctx.device, weights_only=False)
    model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ctx.ddp_local_rank]
        )
    raw_model = unwrap_model(model)

    optimizer = raw_model.configure_optimizer(
        weight_decay=weight_decay, lr=max_lr, device=ctx.device_type
    )

    rm_tokenizer = AutoTokenizer.from_pretrained(RM_NAME)
    rm_model = (
        AutoModelForSequenceClassification.from_pretrained(
            RM_NAME, torch_dtype=torch.bfloat16
        )
        .to(ctx.device)
        .eval()
    )

    ref_model, _ = load_checkpoint(
        checkpoint_file, torch.device("cpu"), weights_only=False
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_dataset = AlpacaPromptDataset(enc_extended, split="train")
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=collate_prompts,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_prompts,
        )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "grpo_finetune.log")

    steps_per_epoch = len(train_loader) // grad_accum_steps
    max_steps = steps_per_epoch * epochs
    eval_every = max(1, max_steps // 170)

    model.eval()

    global_step = 0
    for epoch in range(epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)

        while global_step < max_steps:
            t0 = time.time()

            rollouts = []
            for _ in range(grad_accum_steps):
                try:
                    prompt_batch = next(train_iter).to(ctx.device)
                except StopIteration:
                    break
                _, T_p = prompt_batch.shape
                expanded = prompt_batch.repeat_interleave(group_size, dim=0)

                with torch.no_grad():
                    sequence_ids = raw_model.generate(
                        expanded,
                        max_new_tokens,
                        temperature,
                        top_k,
                        stop_tokens=(IM_END, EOS),
                        pad_token_id=EOS,
                    )
                    action_mask = build_action_mask(sequence_ids, T_p)
                    log_probs_old = compute_sequence_log_probs(
                        model, sequence_ids, action_mask, ctx.device_type
                    )
                    ref_model.to(ctx.device)
                    log_probs_ref = compute_sequence_log_probs(
                        ref_model, sequence_ids, action_mask, ctx.device_type
                    )
                    ref_model.to("cpu")
                    torch.cuda.empty_cache()
                    rewards = compute_rewards(
                        sequence_ids, T_p, enc_extended, rm_model, rm_tokenizer
                    )

                if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                    if ctx.master_process:
                        print(
                            f"step {global_step}: bad rewards in micro-batch, skipping"
                        )
                    continue

                advantages = grpo_advantages(rewards, group_size).to(ctx.device)
                rollouts.append(
                    (
                        sequence_ids,
                        action_mask,
                        log_probs_old,
                        log_probs_ref,
                        advantages,
                        rewards,
                    )
                )

            if not rollouts:
                global_step += 1
                continue

            n_accum = len(rollouts)

            if ctx.master_process and global_step % 100 == 0:
                seq_ids, _, _, _, _, rews = rollouts[0]
                sample_ids = [t for t in seq_ids[0].tolist() if t != EOS]
                sample_text = enc_extended.decode(sample_ids).strip()
                print(
                    "\n[SAMPLE]", f"reward={rews[0].item():.4f}", f"\n{sample_text}\n"
                )

            losses, pg_losses, kl_losses, norms, clip_fracs = [], [], [], [], []
            for _ in range(inner_update_steps):
                loss_accum = pg_loss_accum = kl_loss_accum = clip_frac_accum = 0.0
                optimizer.zero_grad()
                had_nan = False

                for micro_step, (
                    seq_ids,
                    act_mask,
                    lp_old,
                    lp_ref,
                    adv,
                    _,
                ) in enumerate(rollouts):
                    log_probs_new = compute_sequence_log_probs(
                        model, seq_ids, act_mask, ctx.device_type
                    )
                    pg_loss, clip_frac = policy_gradient_loss(
                        log_probs_new,
                        lp_old,
                        adv,
                        act_mask,
                        clip_eps,
                        fixed_len_norm=fixed_len_norm,
                        L_max=max_new_tokens,
                    )
                    kl_loss = kl_penalty_k3(
                        log_probs_new,
                        lp_ref,
                        act_mask,
                        fixed_len_norm=fixed_len_norm,
                        L_max=max_new_tokens,
                    )
                    loss = (pg_loss + beta_kl * kl_loss) / n_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        had_nan = True
                        break

                    if ddp:
                        model.require_backward_grad_sync = micro_step == n_accum - 1
                    loss.backward()
                    loss_accum += loss.detach()
                    pg_loss_accum += pg_loss.item()
                    kl_loss_accum += kl_loss.item()
                    clip_frac_accum += clip_frac

                if had_nan:
                    if ctx.master_process:
                        print(f"step {global_step}: NaN/Inf in inner step, skipping")
                    optimizer.zero_grad()
                    continue

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = get_lr(global_step, max_lr, min_lr, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()

                losses.append(loss_accum.item())
                pg_losses.append(pg_loss_accum / n_accum)
                kl_losses.append(kl_loss_accum / n_accum)
                norms.append(float(norm))
                clip_fracs.append(clip_frac_accum / n_accum)

            t1 = time.time()
            n = max(len(losses), 1)
            avg_loss = sum(losses) / n
            avg_pg = sum(pg_losses) / n
            avg_kl = sum(kl_losses) / n
            avg_norm = sum(norms) / n
            avg_clip = sum(clip_fracs) / max(len(clip_fracs), 1)
            avg_reward = torch.cat([r for *_, r in rollouts]).mean().item()

            if ddp:
                t = torch.tensor(avg_reward, device=ctx.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg_reward = t.item()

            if ctx.master_process:
                print(
                    f"step {global_step:5d} | loss: {avg_loss:.4f} "
                    f"| pg: {avg_pg:.4f} | kl: {avg_kl:.4f} "
                    f"| clip: {avg_clip:.3f} | reward: {avg_reward:.4f} "
                    f"| lr: {lr:.2e} | norm: {avg_norm:.3f} | "
                    f"dt: {(t1 - t0) * 1000:.0f}ms"
                )
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"step {global_step:5d} | loss: {avg_loss:.4f} "
                        f"| pg: {avg_pg:.4f} | kl: {avg_kl:.4f} "
                        f"| clip: {avg_clip:.3f} | reward: {avg_reward:.4f} "
                        f"| lr: {lr:.2e} | norm: {avg_norm:.3f} | "
                        f"dt: {(t1 - t0) * 1000:.0f}ms\n"
                    )
                if (
                    global_step % eval_every == 0 and global_step > 0
                ) or global_step == max_steps - 1:
                    ckpt_path = os.path.join(
                        log_dir, f"grpo_model_{global_step:05d}.pt"
                    )
                    torch.save(
                        {"model": raw_model.state_dict(), "config": raw_model.config},
                        ckpt_path,
                    )

            global_step += 1

    if ddp:
        dist.destroy_process_group()
