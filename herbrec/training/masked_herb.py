import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..data.masking import (
    partial_contexts,
    collate_mhm,
    pair_ranking_loss,
    evaluate_mhm,
)
from ..io import save_arrays, save_checkpoint, save_json, set_seed, sha256
from ..model_loading import load_masked_herb_teacher
from ..teachers.masked_herb import MaskedHerbTeacher
from ..data.candidates import normalize_logits
from ..configuration import MASKED_HERB_CONFIGURATION
from ..data.statistics import frequency_groups


def oof_candidate_data(paths, dataset):
    merged = (
        paths.output / "candidates/oof"
        if (paths.output / "candidates/oof/sample_ids.npy").exists()
        else paths.oof / "merged"
    )
    ids = np.load(merged / "sample_ids.npy")
    full = np.load(merged / "gat_oof_logits.npy")
    folds = np.load(merged / "fold_ids.npy")
    if not np.array_equal(ids, dataset.splits["train"]):
        raise ValueError("OOF candidate IDs differ from frozen training split")
    candidates = np.argsort(-full, axis=1, kind="stable")[:, :50]
    return {
        "sample_ids": ids,
        "candidate_ids": candidates,
        "fold_ids": folds,
        "candidate_ranks": np.broadcast_to(np.arange(1, 51), candidates.shape),
    }


def context_records(dataset, cache, sample_ids, groups, seed_offset):
    lookup = {int(s): i for i, s in enumerate(cache["sample_ids"])}
    labels = dataset.labels(sample_ids)
    rows = []
    for i, sid in enumerate(sample_ids):
        pos = lookup[int(sid)]
        truth = np.flatnonzero(labels[i] > 0).tolist()
        candidates = cache["candidate_ids"][pos]
        contexts = partial_contexts(
            truth,
            candidates,
            cache["candidate_ranks"][pos],
            groups,
            seed_offset + int(sid),
        )
        negatives = [int(h) for h in candidates if labels[i, h] == 0][:8]
        for context in contexts:
            if context["masked"]:
                rows.append(
                    {
                        "sample_id": int(sid),
                        "symptoms": dataset.validate_rows([sid])[0],
                        "visible": context["visible"],
                        "masked": context["masked"],
                        "hard_negatives": negatives,
                    }
                )
    return rows


def train_masked_herb_teacher(
    paths, dataset, fold=0, device_name="cpu", epochs=None, limit=None, max_batches=None
):
    original = paths.mhm / "oof/checkpoints" / f"fold_{fold}"
    from ..io import resolve_device

    device = resolve_device(device_name)
    config = MASKED_HERB_CONFIGURATION.copy()
    set_seed(config["seed"] + fold)
    fit = np.load(original / "fit_sample_ids.npy")
    cal = np.load(original / "calibration_sample_ids.npy")
    heldout = np.load(original / "holdout_sample_ids.npy")
    if np.intersect1d(heldout, np.concatenate([fit, cal])).size:
        raise ValueError("MHM OOF leakage")
    if not np.isin(np.concatenate([fit, cal, heldout]), dataset.splits["train"]).all():
        raise ValueError("MHM fold contains non-training samples")
    full_fit = fit.copy()
    counts = dataset.labels(full_fit).sum(0)
    groups = frequency_groups(counts)
    positive_weights = np.clip(
        (counts.mean() + 1) / np.sqrt(counts + 1), 1, config["positive_weight_cap"]
    ).astype(np.float32)
    if limit is not None:
        fit = fit[:limit]
        cal = cal[:limit]
    cache = oof_candidate_data(paths, dataset)
    train_rows = context_records(dataset, cache, fit, groups, 31000 + fold * 100000)
    cal_rows = context_records(dataset, cache, cal, groups, 41000 + fold * 100000)
    collate = lambda rows: collate_mhm(rows, positive_weights, config)
    train_loader = DataLoader(
        train_rows, batch_size=config["batch_size"], shuffle=True, collate_fn=collate
    )
    cal_loader = DataLoader(
        cal_rows, batch_size=config["eval_batch_size"], collate_fn=collate
    )
    model = MaskedHerbTeacher(
        **{
            key: config[key]
            for key in [
                "hidden_dim",
                "embedding_dim",
                "num_heads",
                "num_set_layers",
                "dropout",
            ]
        }
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    destination = paths.output / "checkpoints/masked_herb_teacher" / f"fold_{fold}"
    best = -1
    stale = 0
    history = []
    for epoch in range(1, (epochs or config["max_epochs"]) + 1):
        model.train()
        losses = []
        for index, batch in enumerate(train_loader):
            if max_batches is not None and index >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["symptoms"].to(device),
                batch["visible"].to(device),
                batch["visible_mask"].to(device),
            )
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, batch["labels"].to(device), reduction="none"
                )
                * batch["weights"].to(device)
            ).sum() / batch["weights"].sum().clamp_min(1).to(device)
            loss += config["lambda_rank"] * pair_ranking_loss(
                logits, batch["positives"], batch["negatives"], config["ranking_margin"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"]
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        score = evaluate_mhm(model, cal_loader, device)["masked_recall_at_10"]
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "masked_recall_at_10": score,
            }
        )
        if score > best:
            best = score
            stale = 0
            save_checkpoint(
                destination / "best_model.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fold": fold,
                    "fit_sample_ids": full_fit,
                    "heldout_sample_ids": heldout,
                    "limited_run": limit is not None or max_batches is not None,
                },
            )
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    np.save(destination / "herb_frequency_groups.npy", groups)
    save_json(destination / "training_history.json", history)
    return destination / "best_model.pt"


def export_masked_herb_targets(paths, dataset, device_name="cpu"):
    cache = oof_candidate_data(paths, dataset)
    lookup = {int(s): i for i, s in enumerate(cache["sample_ids"])}
    logits = np.zeros((len(lookup), 50), np.float32)
    uncertainty = np.zeros_like(logits)
    seen = np.zeros(len(lookup), np.int8)
    sources = []
    for fold in range(5):
        original = paths.mhm / "oof/checkpoints" / f"fold_{fold}"
        directory = paths.output / "checkpoints/masked_herb_teacher" / f"fold_{fold}"
        checkpoint = directory / "best_model.pt"
        if not checkpoint.exists():
            directory = original
            checkpoint = directory / "best_model.pt"
        model, device, payload = load_masked_herb_teacher(checkpoint, device_name)
        if payload.get("limited_run"):
            raise ValueError(
                "Limited training checkpoints cannot produce formal OOF teacher targets"
            )
        fit = payload.get("fit_sample_ids")
        if fit is None:
            fit = np.load(original / "fit_sample_ids.npy")
        heldout = np.load(original / "holdout_sample_ids.npy")
        if np.intersect1d(fit, heldout).size:
            raise ValueError("MHM fit/holdout overlap")
        groups = np.load(directory / "herb_frequency_groups.npy")
        labels = dataset.labels(heldout)
        with torch.inference_mode():
            for row, sid in enumerate(heldout):
                pos = lookup[int(sid)]
                candidates = cache["candidate_ids"][pos]
                truth = np.flatnonzero(labels[row] > 0).tolist()
                contexts = partial_contexts(
                    truth,
                    candidates,
                    cache["candidate_ranks"][pos],
                    groups,
                    51000 + fold * 100000 + int(sid),
                )
                from ..data.masking import pad_herb_sets

                visible, padding, _ = pad_herb_sets(
                    [context["visible"] for context in contexts]
                )
                symptoms = torch.as_tensor(
                    np.repeat(dataset.validate_rows([sid]), len(contexts), axis=0),
                    device=device,
                )
                output = (
                    model(symptoms, visible.to(device), padding.to(device))
                    .cpu()
                    .numpy()[:, candidates]
                )
                valid = np.asarray(
                    [
                        [int(h) not in context["visible"] for h in candidates]
                        for context in contexts
                    ]
                )
                count = valid.sum(0)
                if (count == 0).any():
                    raise ValueError("A candidate has no hidden-context teacher score")
                mean = np.where(valid, output, 0).sum(0) / count
                logits[pos] = mean
                uncertainty[pos] = np.sqrt(
                    np.where(valid, (output - mean) ** 2, 0).sum(0) / count
                )
                seen[pos] += 1
        sources.append({"fold": fold, "checkpoint_sha256": sha256(checkpoint)})
    if not np.all(seen == 1):
        raise ValueError("Incomplete or duplicated MHM OOF predictions")
    output = paths.output / "teacher_signals/masked_herb.npz"
    save_arrays(
        output,
        {
            "sample_ids": cache["sample_ids"],
            "candidate_ids": cache["candidate_ids"],
            "mhm_logits": normalize_logits(logits),
            "mhm_uncertainty": uncertainty,
            "available_mask": np.ones_like(logits),
            "fold_id": cache["fold_ids"],
        },
    )
    save_json(
        paths.output / "teacher_signals/masked_herb_manifest.json",
        {
            "strict_oof": True,
            "validation_used": False,
            "test_used": False,
            "artifact_sha256": sha256(output),
            "checkpoints": sources,
        },
    )
    return output
