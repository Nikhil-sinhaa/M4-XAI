"""
Text faithfulness metrics, ported from InterpretDL evaluate_interpreter/perturbation.py (PerturbationNLP)
and infidelity.py (InfidelityNLP) with the M4 setting percentile=1 (benchmark-nlp/run_eval.sh).
Masked tokens are replaced by [PAD]; the Paddle-style attention mask then drops them (text_models.py).
"""

import numpy as np

from .text_models import TextModel, batched_probs


def perturbation_ids(ids: np.ndarray, expl: np.ndarray, pad_id: int):
    """
    PerturbationNLP.generate_samples(percentile=True): tiles = percentiles 0..100 of expl[1:-1];
    MoRF masks expl >= p for p in tiles[:100] reversed, LeRF masks expl <= p for p in tiles[1:];
    [CLS]/[SEP] are always restored. Each curve: clean sample + 100 perturbed ones.
    """
    tiles = np.percentile(expl[1:-1], list(range(101)))
    cls_id, sep_id = ids[0], ids[-1]

    def masked(sel):
        t = ids.copy()
        t[sel] = pad_id
        t[0], t[-1] = cls_id, sep_id
        return t

    morf = [ids.copy()] + [masked(expl >= p) for p in tiles[:100][::-1]]
    lerf = [ids.copy()] + [masked(expl <= p) for p in tiles[1:]]
    return np.stack(morf), np.stack(lerf)


def morf_lerf_abpc(tm: TextModel, ids, expl, device, bs=16, amp=False):
    ids_np = ids[0].cpu().numpy()
    morf_ids, lerf_ids = perturbation_ids(ids_np, np.asarray(expl), tm.pad_id)
    probs = batched_probs(tm, np.concatenate([morf_ids, lerf_ids]), device, bs, amp)
    coi = int(probs[0].argmax())
    p = probs[:, coi].cpu().numpy()
    morf_p, lerf_p = p[:len(morf_ids)], p[len(morf_ids):]
    return {"MoRF": float(morf_p[0] - morf_p.mean()),
            "LeRF": float(lerf_p[0] - lerf_p.mean()),
            "ABPC": float((lerf_p - morf_p).mean())}, morf_p, lerf_p


class InfidelityCacheNLP:
    """
    InfidelityNLP(is_random_samples=False): [PAD]-mask every window of 1..5 consecutive tokens
    (never [CLS]/[SEP]), I = changed positions, softmax probability of the clean prediction.
    The perturbed probabilities depend only on (model, text), so they are shared by all methods.
    """

    def __init__(self, tm: TextModel, ids, device, bs=16, amp=False):
        base = ids[0].cpu().numpy()
        n = len(base)
        samples = []
        for ks in range(1, 6):
            if ks > n - 2:
                break
            for i in range(1, n - ks):
                t = base.copy()
                t[i:i + ks] = tm.pad_id
                samples.append(t)
        samples = np.stack(samples)
        self.I = (samples != base[None, :]).astype(np.float64)
        p0 = batched_probs(tm, base[None], device, 1, amp)[0]
        self.label = int(p0.argmax())
        pert = batched_probs(tm, samples, device, bs, amp)[:, self.label]
        self.diff = (p0[self.label] - pert).cpu().numpy().astype(np.float64)

    def score(self, expl) -> float:
        exp_sum = self.I @ np.asarray(expl, dtype=np.float64)
        denom = np.mean(exp_sum * exp_sum)
        exp_sum = np.zeros_like(exp_sum) if denom == 0.0 else exp_sum * (np.mean(self.diff * exp_sum) / denom)
        return float(np.mean(np.square(self.diff - exp_sum)))
