"""Active Learning training and experimentation code."""
import traceback
import logging
from typing import Tuple

import h5py
import numpy as np
import torch
import os
from numpy.typing import NDArray
from omegaconf import DictConfig

from ALFM.src.classifiers.classifier_wrapper import ClassifierWrapper
from ALFM.src.clustering.label_prop import LabelPropagation
from ALFM.src.init_strategies.registry import InitType
from ALFM.src.query_strategies.registry import QueryType
from ALFM.src.run.utils import ExperimentLogger
from h5py import h5d, h5t, h5s
from ALFM.compute_logging import WallTimer, append_jsonl, linear_head_flops_estimate

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    """Fix the NumPy and PyTorch seeds.

    Args:
        seed: the value of the random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _safe_read_float_dataset(dset: h5py.Dataset, out_dtype=np.float32) -> np.ndarray:
    """
    Read an HDF5 float dataset even when h5py can't map the on-disk float type
    (e.g., custom 16-bit floats like bfloat16). Prefer HDF5's own conversion into
    a float32 buffer; otherwise fall back to raw-bytes conversion.
    """
    # 1) Try normal high-level read
    try:
        arr = dset[()]
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
            return arr.astype(out_dtype, copy=False)
        return np.asarray(arr, dtype=out_dtype)
    except ValueError as e:
        if "Insufficient precision" not in str(e):
            raise

    shape = dset.shape
    sid = dset.id

    # 2) Ask HDF5 to convert file float -> float32 directly into our buffer
    out = np.empty(shape, dtype=np.float32 if out_dtype == np.float32 else np.float64)
    try:
        # h5py API variant: read(mspace, fspace, buf)
        sid.read(h5s.ALL, h5s.ALL, out)
        return out.astype(out_dtype, copy=False)
    except TypeError:
        pass
    try:
        # h5py API variant: read(mspace, fspace, buf=...)
        sid.read(h5s.ALL, h5s.ALL, buf=out)
        return out.astype(out_dtype, copy=False)
    except TypeError:
        pass
    try:
        # h5py API variant: read(mtype, mspace, fspace, dxpl, buf)
        mtype = h5t.py_create(np.dtype(out.dtype))
        sid.read(mtype, h5s.ALL, h5s.ALL, None, out)
        return out.astype(out_dtype, copy=False)
    except TypeError:
        try:
            # Some builds omit dxpl arg
            mtype = h5t.py_create(np.dtype(out.dtype))
            sid.read(mtype, h5s.ALL, h5s.ALL, out)
            return out.astype(out_dtype, copy=False)
        except Exception:
            pass
    except Exception:
        # fall through to raw-bytes path
        pass

    # 3) Raw-bytes path: copy bits and interpret them ourselves
    try:
        ftype = sid.get_type()
        sz = ftype.get_size()  # bytes per element
        total = int(np.prod(shape)) * sz
        buf = bytearray(total)

        # Try read into Python buffer; test both signatures
        try:
            # mtype = file type -> raw copy
            sid.read(ftype, h5s.ALL, h5s.ALL, None, buf)
        except TypeError:
            sid.read(ftype, h5s.ALL, h5s.ALL, buf)

        # Interpret bytes
        raw = memoryview(buf)
        if sz == 2:
            # Try float16 and bfloat16; pick the saner one
            f16 = np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            bf32 = np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
            def score(a: np.ndarray) -> float:
                a = a.astype(np.float32, copy=False)
                fin = np.isfinite(a).mean()
                nz = (np.abs(a) > 0).mean()
                return float(fin + 0.1 * nz)
            picked = f16 if score(f16) >= score(bf32) else bf32
            return picked.reshape(shape).astype(out_dtype, copy=False)
        elif sz == 4:
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(out_dtype, copy=False)
        elif sz == 8:
            return np.frombuffer(raw, dtype=np.float64).reshape(shape).astype(out_dtype, copy=False)
        else:
            raise ValueError(f"Unsupported float type size in HDF5: {sz} bytes")
    except Exception as ex:
        raise RuntimeError(f"Failed to read dataset '{dset.name}' with safe reader: {ex}") from ex

def _safe_read_int_dataset(dset: h5py.Dataset, out_dtype=np.int64) -> np.ndarray:
    arr = dset[()]
    return np.asarray(arr, dtype=out_dtype)

def load_vectors(vector_file: str):
    with h5py.File(vector_file, "r") as fh:
        train_x = _safe_read_float_dataset(fh["train/features"], np.float32)
        train_y = np.asarray(fh["train/labels"][()], dtype=np.int64)
        test_x  = _safe_read_float_dataset(fh["test/features"],  np.float32)
        test_y  = np.asarray(fh["test/labels"][()],  dtype=np.int64)
    return train_x, train_y, test_x, test_y

def al_train(vector_file: str, log_dir: str, cfg: DictConfig) -> None:
    exp_logger = ExperimentLogger(log_dir, cfg)
    set_seed(cfg.seed)
    train_x, train_y, test_x, test_y = load_vectors(vector_file)

    num_classes = cfg.dataset.num_classes
    budget_step = cfg.budget.step * num_classes
    budget_init = cfg.budget.init * num_classes

    iterations = np.arange(1, cfg.iterations.n + 1)
    if cfg.iterations.exp:
        iterations = 2 ** (iterations - 1)

    # ---- build strategies (init + query) ----
    init_strategy = InitType[cfg.init_strategy.name]
    init_sampler = init_strategy.value(
        features=train_x, labels=train_y, **cfg.init_strategy.params
    )

    query_strategy = QueryType[cfg.query_strategy.name]
    query_sampler = query_strategy.value(
        features=train_x,
        labels=train_y,
        init_sampler=init_sampler,
        **cfg.query_strategy.params,
    )

    # ---- WARM START FIRST (before SSL, before update_state, before train) ----
    N = len(train_x)
    labeled_pool = np.zeros(N, dtype=bool)

    try:
        if hasattr(query_sampler, "initial_seed"):
            # IMPORTANT: do NOT call update_state yet; there is no labeled pool.
            seed_mask = query_sampler.initial_seed(budget_init)  # centroid seeding in CLIP
            assert isinstance(seed_mask, np.ndarray) and seed_mask.dtype == bool and seed_mask.size == N, \
                f"initial_seed must return a bool mask of size N, got dtype={getattr(seed_mask,'dtype',None)} shape={getattr(seed_mask,'shape',None)}"
            labeled_pool = seed_mask.copy()
            logging.getLogger(__name__).info(
                "[Runner] Warm-start enabled: seeded %d points via strategy.initial_seed()",
                int(labeled_pool.sum()),
            )
        else:
            # fallback: configured init sampler (e.g., random)
            seed_mask = init_sampler.query(budget_init)
            assert isinstance(seed_mask, np.ndarray) and seed_mask.dtype == bool and seed_mask.size == N, \
                f"init_sampler must return a bool mask of size N, got dtype={getattr(seed_mask,'dtype',None)} shape={getattr(seed_mask,'shape',None)}"
            labeled_pool = seed_mask.copy()
            logging.getLogger(__name__).info(
                "[Runner] Warm-start not available; using init_sampler to seed %d points",
                int(labeled_pool.sum()),
            )
    except Exception as e:
        logging.getLogger(__name__).error("Warm-start seeding failed: %s\n%s", e, traceback.format_exc())
        raise

    # ---- now we may safely fit SSL and update the strategy state ----
    ssl = LabelPropagation(**cfg.ssl)
    try:
        ssl.fit(train_x)  # harmless when ssl.enable=False
    except Exception as e:
        logging.getLogger(__name__).error("SSL fit failed: %s\n%s", e, traceback.format_exc())
        raise

    # first state ingest AFTER we have labels
    try:
        query_sampler.update_state(iteration=0, labeled_pool=labeled_pool, model=None)
    except Exception as e:
        logging.getLogger(__name__).error("query_sampler.update_state(iter=0) failed: %s\n%s", e, traceback.format_exc())
        raise

    pending_ccma =None
    # ------------------- ACTIVE LEARNING LOOP -------------------
    for i, iteration in enumerate(iterations, 1):
        # sanity log (useful to catch pool size regressions)
        logging.getLogger(__name__).info("[Runner] Iteration %d/%d: labeled=%d",
                                         i, len(iterations), int(labeled_pool.sum()))

        # semi-supervised pseudo-labels (works even if disabled)
        ssl_y = ssl.predict(train_y, labeled_pool)

        # train
        model = ClassifierWrapper(cfg)
        # Add wall time logging around model.fit
        # with WallTimer() as t_train:
        model.fit(train_x, train_y, labeled_pool, ssl_y)

        # eval
        # with WallTimer() as t_eval:
        scores = model.eval(test_x, test_y)
        exp_logger.log_scores(scores, i, len(iterations), labeled_pool.sum(), extras=pending_ccma)
        # # log timing info
        # timing_row = {
        #     "seed": int(cfg.seed),
        #     "variant": str(cfg.query_strategy.name),    # e.g., 'disagreement_V1'
        #     "iteration": int(i),
        #     "labeled": int(labeled_pool.sum()),
        #     "num_classes": int(cfg.dataset.num_classes),
        #     "epochs": int(getattr(cfg.trainer, "max_epochs", 1)),
        #     "batch_size": int(getattr(cfg.dataloader, "batch_size", 128)),
        #     "t_train_sec": float(t_train.dt),
        #     "t_eval_sec": float(t_eval.dt),
        #     # if you have query totals per round, add it here; else leave 0
        #     "t_query_total_sec": 0.0,  # or read from your timing CSV if available
        # }
        # # --- Add FLOPs estimate for linear head ---
        # D = model.num_features               # student feature dimension
        # C = cfg.dataset.num_classes
        # N_lab = int(labeled_pool.sum())      # number of labeled samples this round
        # E = int(getattr(cfg.trainer, "max_epochs", 1))
        # flops = linear_head_flops_estimate(N_lab, D, C, E)
        # timing_row["linear_head_flops_est"] = flops

        # append_jsonl(os.path.join(log_dir, "compute_iter_timing.jsonl"), timing_row)

        if i == len(iterations):
            return  # done

        # compute budget for the NEXT increment
        if cfg.iterations.exp:
            next_units = iterations[i]
            prev_units = iterations[i - 1]
            budget = (next_units - prev_units) * num_classes * cfg.budget.step
        else:
            budget = budget_step

        # update state with trained model
        query_sampler.update_state(iteration, labeled_pool, model)

        # query new points (expects bool mask over N)
        indices_mask = query_sampler.query(budget)

        # pending_ccma = getattr(query_sampler, "last_ccma", None)
        
        assert isinstance(indices_mask, np.ndarray) and indices_mask.dtype == bool and indices_mask.size == N, \
            f"query() must return a bool mask of size N, got dtype={getattr(indices_mask,'dtype',None)} shape={getattr(indices_mask,'shape',None)}"

        # only add newly unlabeled
        add_mask = indices_mask & (~labeled_pool)
        if add_mask.sum() < indices_mask.sum():
            logging.getLogger(__name__).warning(
                "[Runner] Query returned %d already-labeled; adding %d new.",
                int(indices_mask.sum() - add_mask.sum()),
                int(add_mask.sum()),
            )

        labeled_pool |= add_mask




