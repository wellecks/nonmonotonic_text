import torch
import numpy as np
import torch.nn.functional as F

import tree_text_gen.binary.common.util as util


class GreedySampler(object):
    def __init__(self, aux_end=False, end_idx=4):
        self.aux_end = aux_end
        self.end_idx = end_idx

    def __call__(self, scores, **kwargs):
        if self.aux_end:
            token_scores, stop_probs = scores
            tokens = token_scores.argmax(2)
            stop = (stop_probs >= 0.5).squeeze(1)
            # If stop was predicted, put the <end> symbol instead of the token
            tokens[stop, :] = self.end_idx
            return tokens
        else:
            return scores.argmax(2)


class StochasticSampler(object):
    def __init__(self, aux_end=False, end_idx=4):
        self.aux_end = aux_end
        self.end_idx = end_idx

    def __call__(self, scores, **kwargs):
        if self.aux_end:
            token_scores, stop_probs = scores
            token_ps = F.softmax(token_scores.squeeze(1), dim=1)
            tokens = token_ps.multinomial(num_samples=1)
            stop = (stop_probs >= 0.5).squeeze(1)
            # If stop was predicted, put the <end> symbol instead of the token
            tokens[stop, :] = self.end_idx
            return tokens
        else:
            scores = scores.squeeze(1)
            ps = F.softmax(scores, dim=1)
            samples = ps.multinomial(num_samples=1)
        return samples


class MultinomialSampler(object):
    def __init__(self, aux_end=False, end_idx=4):
        self.aux_end = aux_end
        self.end_idx = end_idx

    def __call__(self, ps, **kwargs):
        if self.aux_end:
            token_ps, stop_ps = ps
            samples = token_ps.squeeze(1).multinomial(num_samples=1).detach()
            stop = (stop_ps >= 0.5).squeeze(1)  # Just predict, instead of sample, stop
            # If stop was predicted, put the <end> symbol instead of the token
            samples[stop, :] = self.end_idx
        else:
            ps = ps.squeeze(1)
            samples = ps.multinomial(num_samples=1).detach()
        return samples


class TopkSampler(object):
    def __init__(self, k, device, aux_end=False):
        self.device = device
        self.k = k
        self.aux_end = aux_end
        if self.aux_end:
            raise NotImplementedError('')

    def __call__(self, scores, **kwargs):
        with torch.no_grad():
            scores = scores.squeeze(1)
            k = min(self.k, scores.size(1))
            topk_scores, topk_idxs = scores.topk(k)
            sampled_idxs = F.softmax(topk_scores, 1).multinomial(1)
            samples = topk_idxs.gather(1, sampled_idxs)
        return samples


class PolicyCorrectSampler(object):
    """A training sampler that samples from a distribution proportional to the policy's
       distribution restricted to correct actions."""
    def __init__(self, sample_dim, device, base_sampler):
        self.base_sampler = base_sampler
        self.device = device
        self.sample_dim = sample_dim
        self.vmax = int(np.sqrt(sample_dim-1))
        self.aux_end = base_sampler.aux_end
        self.end_idx = base_sampler.end_idx

    def __call__(self, scores, oracle, **kwargs):
        with torch.no_grad():
            if self.aux_end:
                token_scores, stop_probs = scores
                p_oracle = oracle.distribution()
                correct_actions_mask = p_oracle.gt(0).unsqueeze(1).float()
                stop_probs = correct_actions_mask[:, :, self.end_idx]  # prevent invalid <end>
                token_scores = torch.clamp(token_scores, -40, 40)
                ps = util.masked_softmax(token_scores, correct_actions_mask, dim=2)
                samples = self.base_sampler((ps, stop_probs))
            else:
                p_oracle = oracle.distribution()
                correct_actions_mask = p_oracle.gt(0).unsqueeze(1).float()
                scores = torch.clamp(scores, -40, 40)
                ps = util.masked_softmax(scores, correct_actions_mask, dim=2)
                samples = self.base_sampler(ps)
        return samples


class MixedRollin(object):
    def __init__(self, beta, training_sampler, eval_sampler, mix_type=None):
        """Higher beta => uses oracle more often."""
        self.beta = beta
        self.use_oracle = False
        self.training_sampler = training_sampler
        self.eval_sampler = eval_sampler
        if mix_type not in ['trajectory', 'state']:
            raise Exception("Unknown rollin mix type: " + str(mix_type))
        self.mix_type = mix_type

    def reset(self, **kwargs):
        self.use_oracle = np.random.binomial(1, p=self.beta) > 0

    def __call__(self, scores, oracle, training, **kwargs):
        if self.mix_type == 'state':
            self.reset()
        if self.use_oracle and oracle is not None and training:
            samples = oracle.sample()
        elif training:
            samples = self.training_sampler(scores, oracle=oracle, **kwargs)
        else:
            samples = self.eval_sampler(scores, **kwargs)
        return samples


def initialize(args):
    """Helper method to hide rollin and sampler initialization from the training script."""
    if args.training_sampler == 'policy_correct_greedy':
        training_sampler = PolicyCorrectSampler(args.n_classes, args.device, base_sampler=GreedySampler(aux_end=args.aux_end))
    elif args.training_sampler == 'policy_correct_stochastic':
        training_sampler = PolicyCorrectSampler(args.n_classes, args.device, base_sampler=MultinomialSampler(aux_end=args.aux_end))
    elif args.training_sampler == 'stochastic':
        training_sampler = StochasticSampler(args.aux_end)
    elif args.training_sampler == 'greedy':
        training_sampler = GreedySampler(args.aux_end)
    else:
        raise NotImplementedError("training sampler %s" % args.training_sampler)

    if args.eval_sampler == 'stochastic':
        eval_sampler = StochasticSampler(args.aux_end)
    elif args.eval_sampler == 'greedy':
        eval_sampler = GreedySampler(args.aux_end)
    else:
        raise NotImplementedError("eval sampler %s" % args.eval_sampler)

    if args.rollin == 'mixed':
        rollin = MixedRollin(args.rollin_beta, training_sampler, eval_sampler, args.rollin_mix_type)
    else:
        raise NotImplementedError("rollin %s" % args.rollin)

    return rollin
