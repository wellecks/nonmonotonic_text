"""Support non-incremental updates, used when oracle sampling for the Transformer."""
import torch

class TrajectoryBatch(object):
    def __init__(self, x, y, p_oracle):
        self.x = x
        self.y = y
        self.p_oracle = p_oracle

class TrajectorySampler(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = dataloader.__iter__()

    def get_batch(self):
        batch = next(self.iterator)
        if batch is None:
            self.iterator = self.dataloader.__iter__()
            batch = next(self.iterator)
        return batch

    def get_mixed_trajectory_loss(self, model, oracle_class, oracle_flags, **kwargs):
        batch = self.get_batch()
        loss = model(xs=batch.src, ys=batch.trg[0], p_oracle=None, oracle_cls=oracle_class,
                     oracle_flags=oracle_flags, max_steps=batch.trg[0].size(1)*2+1, **kwargs)
        loss = loss.mean()
        # Needs to be set outside of forward (see warning in `DataParallel`)
        model.module.longest_label = max(model.module.longest_label, batch.trg[0].size(1)*2+1)
        return loss

    def get_oracle_trajectory(self, model, oracle_class, oracle_flags):
        batch = self.get_batch()
        oracle = oracle_class(batch.trg[0].detach(), model.n_classes, model.tok2i, model.i2tok, **oracle_flags)
        ps, ys = [], []
        while not oracle.done():
            pt, yt = oracle.distribution(), oracle.sample()
            oracle.update(yt)
            ps.append(pt)
            ys.append(yt)
        return TrajectoryBatch(batch.src, torch.cat(ys, 1), torch.stack(ps, 1))

    def get_loss(self, model, trajectory_batch, loss_flags):
        loss = model(xs=trajectory_batch.x, ys=trajectory_batch.y, p_oracle=trajectory_batch.p_oracle, num_samples=trajectory_batch.y.size(0), loss_flags=loss_flags)
        loss = loss.mean()
        # Needs to be set outside of forward (see warning in `DataParallel`)
        model.module.longest_label = max(model.module.longest_label, trajectory_batch.y.size(1))
        return loss


