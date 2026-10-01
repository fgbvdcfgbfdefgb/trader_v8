"""Minimal PPO trainer for the TradeMaker actor-critic."""
import numpy as np
import torch
import torch.nn.functional as F


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.logprobs, self.values, self.rewards, self.dones = (
            [], [], [], [], [], []
        )

    def add(self, obs, action, logprob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:-1], dtype=np.float32)
    return advantages, returns


class PPOTrainer:
    def __init__(self, model, device, lr=3e-4, clip=0.2, epochs=4, batch_size=256,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.model(obs)
            dist = torch.distributions.Normal(mu, std)
            raw_action = dist.sample()
            logprob = dist.log_prob(raw_action).sum(-1)
            weights = F.softmax(raw_action, dim=-1)
        return (
            weights.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            float(logprob.item()),
            float(value.item()),
        )

    def update(self, buf: RolloutBuffer, last_value=0.0):
        obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(buf.actions), dtype=torch.float32, device=self.device)
        old_logprobs = torch.as_tensor(np.array(buf.logprobs), dtype=torch.float32, device=self.device)

        advantages, returns = compute_gae(buf.rewards, buf.values, buf.dones, last_value=last_value)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(buf)
        idx = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                b_obs = obs[b]
                b_actions = actions[b]
                b_old_logprobs = old_logprobs[b]
                b_adv = advantages[b]
                b_ret = returns[b]

                mu, std, value = self.model(b_obs)
                dist = torch.distributions.Normal(mu, std)
                logprob = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(logprob - b_old_logprobs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, b_ret)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats
