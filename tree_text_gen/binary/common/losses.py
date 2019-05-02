import torch as th
import torch.nn.functional as F

import tree_text_gen.binary.common.util as util


def _correct_policy_distribution(scores_t, p_oracle):
    # make a renormalized distribution using the policy's probabilities over the correct set of actions
    # (and setting the probability of other actions to zero)
    scores_t = scores_t.clamp(-40, 40)  # for numerical stability
    correct_actions_mask = p_oracle.gt(0).float()
    p_correct_policy = util.masked_softmax(scores_t, correct_actions_mask, dim=1)
    return p_correct_policy


def sequential_set_no_stop_loss(scores, samples, oracle_ps, end_idx, self_teach_beta=1.0):
    T = scores.size(1)
    losses = []
    for t in range(T):
        p_oracle = oracle_ps[:, t, :]

        # annealed coaching
        if self_teach_beta < 1.0:
            p_correct_policy = _correct_policy_distribution(scores[:, t, :], p_oracle)
            p_oracle = (1.0 - self_teach_beta)*p_correct_policy + self_teach_beta*p_oracle
            p_oracle = (p_oracle / p_oracle.sum(1, keepdim=True)).detach()

        logp_policy = F.log_softmax(scores[:, t, :], dim=1)
        loss_ = F.kl_div(logp_policy, p_oracle, reduction='none').sum(1)

        losses.append(loss_)

    # Generation ended when the number of end labels exceeds the number of labels
    end_indicator = ((samples == end_idx).cumsum(1) > (samples != end_idx).cumsum(1))
    # 1, 2, ..., t = 1 where t is the end time, t+1,...,T = 0.
    end_mask = (end_indicator.cumsum(1).cumsum(1) <= 1).float()

    # One word plus two end tokens at minimum.
    min_steps = 3
    end_mask[:, :min_steps] = 1

    losses = th.stack(losses, 1)
    loss = ((losses*end_mask).sum(1) / th.clamp(end_mask.sum(1), min=1.0)).mean()
    loss = loss
    return loss


def sequential_set_loss(scores, samples, oracle_ps, end_idx, self_teach_beta=1.0):
    # -- Version with auxiliary <end> loss
    if not isinstance(scores[0], tuple) and scores[0].size(1) > 1:
        scores, stop_probs = scores
    else:
        scores, stop_probs = th.cat([s[0] for s in scores], 1), th.cat([s[1] for s in scores], 1)
    T = scores.size(1)
    losses = []
    stop_losses = []
    for t in range(T):
        if isinstance(oracle_ps, list):
            p_oracle = oracle_ps[t]
        else:
            p_oracle = oracle_ps[:, t, :]

        # annealed coaching
        if self_teach_beta < 1.0:
            p_correct_policy = _correct_policy_distribution(scores[:, t, :], p_oracle)
            p_oracle = (1.0 - self_teach_beta)*p_correct_policy + self_teach_beta*p_oracle
            p_oracle = (p_oracle / p_oracle.sum(1, keepdim=True)).detach()

        logp_policy = F.log_softmax(scores[:, t, :], dim=1)
        loss_ = F.kl_div(logp_policy, p_oracle, reduction='none').sum(1)

        # <end> loss
        stop_oracle = p_oracle[:, end_idx]
        stop_loss = F.binary_cross_entropy(stop_probs[:, t], stop_oracle, reduction='none')

        losses.append(loss_)
        stop_losses.append(stop_loss)

    # Generation ended when the number of end labels exceeds the number of labels
    end_indicator = ((samples == end_idx).cumsum(1) > (samples != end_idx).cumsum(1))
    # 1, 2, ..., t = 1 where t is the end time, t+1,...,T = 0.
    end_mask = (end_indicator.cumsum(1).cumsum(1) <= 1).float()

    # 0 for <end>, 1 otherwise
    tokens_mask = (samples != end_idx).float()

    # One word plus two end tokens at minimum.
    min_steps = 3
    end_mask[:, :min_steps] = 1

    # for token loss, mask steps after completion, and <end> steps (which will have an <end> loss instead)
    token_loss_mask = end_mask*tokens_mask

    # for stop loss, mask steps after completion
    stop_loss_mask = end_mask

    losses = th.stack(losses, 1)
    loss = ((losses*token_loss_mask).sum(1) / th.clamp(token_loss_mask.sum(1), min=1.0)).mean()

    stop_losses = th.stack(stop_losses, 1)
    stop_loss = ((stop_losses*stop_loss_mask).sum(1) / th.clamp(stop_loss_mask.sum(1), min=1.0)).mean()

    loss = loss + stop_loss
    return loss

