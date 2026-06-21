import os
import json
import csv
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import signal
import sys

TRAIN_DIR   = "results_coreset_selection/coresets/hdbscan_kmeans"
VAL_DIR     = "apple_dataset/val"
TEST_DIR    = "apple_dataset/test"

CLASS_NAMES = [
    "apple_frogeye_leaf_spot", "apple_leaf_healthy",      "apple_mosaic_leaf",
    "apple_powdery_mildew_leaf", "apple_rust_leaf",       "apple_scab_leaf",
]

IMG_SIZE    = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)

EPOCHS      = 40
PATIENCE    = 10

SEEDS           = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 100
POP_SIZE        = 10
N_GENERATIONS   = 10

PSO_W  = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

RESULTS_DIR = Path("results_pso")

EXTERNAL_SUMMARY_CSVS = [
    "results_ga/summary.csv",
    "results_de/summary.csv",
]

SEARCH_SPACE = {
    "freezing_ratio": [0.70, 0.80, 0.85, 0.90, 0.95, 0.99],
    "learning_rate":  [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "dropout_rate":   [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    "l2_reg":         [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    "optimizer":      ["adam", "adamw", "sgd"],
    "batch_size":     [8, 16, 32],
}

HP_KEYS   = list(SEARCH_SPACE.keys())
HP_VALUES = [SEARCH_SPACE[k] for k in HP_KEYS]
DIM       = len(HP_KEYS)

BOUNDS    = np.array([len(v) for v in HP_VALUES], dtype=float)
V_MAX     = BOUNDS * 0.20

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n  [Signal] Ctrl+C / SIGTERM detected — finishing current trial "
          "then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def vec_to_indices(vec: np.ndarray) -> np.ndarray:
    clipped = np.clip(vec, 0.0, BOUNDS - 1e-9)
    return np.floor(clipped).astype(int)

def indices_to_dict(idx: np.ndarray) -> dict:
    return {k: HP_VALUES[i][idx[i]] for i, k in enumerate(HP_KEYS)}

def vec_to_dict(vec: np.ndarray) -> dict:
    return indices_to_dict(vec_to_indices(vec))

def config_hash(hp_dict: dict) -> str:
    s = json.dumps(hp_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]

def random_vector(rng_np: np.random.Generator) -> np.ndarray:
    return rng_np.uniform(0.0, BOUNDS)

def pso_update_velocity(vel: np.ndarray,
                        pos: np.ndarray,
                        pbest: np.ndarray,
                        gbest: np.ndarray,
                        rng_np: np.random.Generator) -> np.ndarray:
    r1 = rng_np.random(DIM)
    r2 = rng_np.random(DIM)
    new_vel = (PSO_W  * vel
               + PSO_C1 * r1 * (pbest - pos)
               + PSO_C2 * r2 * (gbest - pos))
    return np.clip(new_vel, -V_MAX, V_MAX)

def pso_update_position(pos: np.ndarray,
                        vel: np.ndarray) -> np.ndarray:
    new_pos = pos + vel
    return np.clip(new_pos, 0.0, BOUNDS - 1e-9)

def seed_dir(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}"

def trial_path(seed: int, trial_id: int) -> Path:
    return seed_dir(seed) / "trials" / f"trial_{trial_id:03d}.json"

def checkpoint_path(seed: int) -> Path:
    return seed_dir(seed) / "pso_checkpoint.json"

def summary_csv_path() -> Path:
    return RESULTS_DIR / "summary.csv"

def setup_dirs(seed: int):
    (seed_dir(seed) / "trials").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_one_csv(path: Path, cache: dict, label: str) -> int:
    if not path.exists():
        print(f"  [Cache] Skipping '{path}' — file not found.")
        return 0

    added = 0
    try:
        df = pd.read_csv(path)
        available_keys = [k for k in HP_KEYS if k in df.columns]
        if len(available_keys) < len(HP_KEYS):
            missing = set(HP_KEYS) - set(available_keys)
            print(f"  [Cache] Warning: '{path}' is missing columns {missing}.")

        for _, row in df.iterrows():
            try:
                hp = {}
                for k in HP_KEYS:
                    val = row[k]
                    if pd.isna(val):
                        raise ValueError(f"NaN for key {k}")
                    expected = SEARCH_SPACE[k][0]
                    if isinstance(expected, int):
                        val = int(val)
                    elif isinstance(expected, float):
                        val = float(val)
                    hp[k] = val
            except (KeyError, ValueError):
                continue

            h        = config_hash(hp)
            val_acc  = float(row["val_accuracy"])  if pd.notna(row.get("val_accuracy"))  else None
            test_acc = float(row["test_accuracy"]) if pd.notna(row.get("test_accuracy")) else None

            if val_acc is None:
                continue

            if h not in cache or val_acc > cache[h]["val_accuracy"]:
                cache[h] = {
                    "val_accuracy":  val_acc,
                    "test_accuracy": test_acc,
                    "source":        label,
                }
                added += 1

    except Exception as e:
        print(f"  [Cache] Warning: could not load '{path}' — {e}")

    return added

def build_cache_from_csv() -> dict:
    cache: dict = {}
    total_external = 0

    if EXTERNAL_SUMMARY_CSVS:
        print(f"  [Cache] Loading {len(EXTERNAL_SUMMARY_CSVS)} external "
              f"summary file(s)…")
        for raw_path in EXTERNAL_SUMMARY_CSVS:
            p = Path(raw_path)
            n = _load_one_csv(p, cache, label=p.name)
            total_external += n
            print(f"           {p}  →  {n} new entries")
    else:
        print("  [Cache] No external summary files configured.")

    own_p = summary_csv_path()
    n_own = _load_one_csv(own_p, cache, label="pso_summary.csv")
    if n_own:
        print(f"  [Cache] PSO own summary ({own_p})  →  {n_own} new/updated entries")

    print(f"  [Cache] Total unique configs in cache: {len(cache)}  "
          f"(external: {total_external}, PSO own: {n_own})")
    return cache

def save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_epochs, elapsed,
               from_cache=False):
    log = {
        "method":                "particle_swarm_optimization",
        "seed":                  seed,
        "trial_id":              trial_id,
        "hyperparams":           hyperparams,
        "val_accuracy":          float(val_acc),
        "test_accuracy":         float(test_acc) if test_acc is not None else None,
        "val_loss":              (float(min(history_dict["val_loss"]))
                                  if history_dict else None),
        "best_epoch":            int(best_epoch)   if best_epoch   is not None else None,
        "total_epochs":          int(total_epochs) if total_epochs is not None else None,
        "history":               history_dict,
        "training_time_seconds": round(elapsed, 1),
        "from_cache":            from_cache,
        "timestamp":             datetime.now().isoformat(),
    }
    with open(trial_path(seed, trial_id), "w") as f:
        json.dump(log, f, indent=2)

    csv_p        = summary_csv_path()
    write_header = not csv_p.exists()
    with open(csv_p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "seed", "trial_id",
            *HP_KEYS,
            "val_accuracy", "test_accuracy", "best_epoch",
            "total_epochs", "training_time_seconds", "from_cache", "timestamp",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method":                "particle_swarm_optimization",
            "seed":                  seed,
            "trial_id":              trial_id,
            **hyperparams,
            "val_accuracy":          log["val_accuracy"],
            "test_accuracy":         log["test_accuracy"],
            "best_epoch":            log["best_epoch"],
            "total_epochs":          log["total_epochs"],
            "training_time_seconds": round(elapsed, 1),
            "from_cache":            from_cache,
            "timestamp":             log["timestamp"],
        })

def load_trial(seed: int, trial_id: int):
    p = trial_path(seed, trial_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def save_checkpoint(seed, generation,
                    positions, velocities, fitnesses,
                    pbest_positions, pbest_fitnesses,
                    evaluated_hashes, trial_counter,
                    best_val_acc, best_hyperparams,
                    rng_np: np.random.Generator):
    ckpt = {
        "generation":       generation,
        "positions":        [v.tolist() for v in positions],
        "velocities":       [v.tolist() for v in velocities],
        "fitnesses":        list(fitnesses),
        "pbest_positions":  [p.tolist() for p in pbest_positions],
        "pbest_fitnesses":  list(pbest_fitnesses),
        "evaluated_hashes": list(evaluated_hashes),
        "trial_counter":    trial_counter,
        "best_val_acc":     best_val_acc,
        "best_hyperparams": best_hyperparams,
        "rng_state":        rng_np.bit_generator.state,
        "timestamp":        datetime.now().isoformat(),
    }
    p   = checkpoint_path(seed)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, p)
    print(f"  [Checkpoint] seed={seed} gen={generation} "
          f"trial={trial_counter}/{MAX_EVALUATIONS} "
          f"best={best_val_acc:.4f}")

def load_checkpoint(seed: int):
    p = checkpoint_path(seed)
    if not p.exists():
        return None, np.random.default_rng(seed)

    with open(p) as f:
        ckpt = json.load(f)

    ckpt["positions"]       = [np.array(v) for v in ckpt["positions"]]
    ckpt["velocities"]      = [np.array(v) for v in ckpt["velocities"]]
    ckpt["pbest_positions"] = [np.array(v) for v in ckpt["pbest_positions"]]
    ckpt["evaluated_hashes"] = set(ckpt["evaluated_hashes"])

    rng_np = np.random.default_rng(seed)
    rng_np.bit_generator.state = ckpt["rng_state"]

    print(f"  [Checkpoint loaded] seed={seed} "
          f"gen={ckpt['generation']} "
          f"trials={ckpt['trial_counter']}/{MAX_EVALUATIONS} "
          f"best={ckpt['best_val_acc']:.4f} "
          f"best_hp={ckpt.get('best_hyperparams')}")
    return ckpt, rng_np

def load_datasets(batch_size: int):
    import tensorflow as tf

    def make_ds(directory, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42,
        )

    train_ds = make_ds(TRAIN_DIR, shuffle=True)
    val_ds   = make_ds(VAL_DIR,   shuffle=False)
    test_ds  = make_ds(TEST_DIR,  shuffle=False)

    preprocess = tf.keras.applications.efficientnet.preprocess_input
    AUTOTUNE   = tf.data.AUTOTUNE

    train_ds = (train_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds   = (val_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    test_ds  = (test_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    return train_ds, val_ds, test_ds

def build_model(freezing_ratio: float, dropout_rate: float,
                l2_reg: float, num_classes: int = NUM_CLASSES):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    base_model = keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    total_layers = len(base_model.layers)
    fine_tune_at = int(freezing_ratio * total_layers)

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    return keras.Model(inputs, outputs, name="AppleLeaf_EfficientNetB0")

def train_and_eval(hyperparams: dict, seed: int, trial_id: int) -> tuple:
    import tensorflow as tf
    from tensorflow import keras

    trial_seed = seed * 1000 + trial_id
    tf.random.set_seed(trial_seed)
    np.random.seed(trial_seed)

    hp = hyperparams
    t0 = time.time()

    train_ds_hp, val_ds_hp, test_ds_hp = load_datasets(hp["batch_size"])

    model = build_model(
        freezing_ratio=hp["freezing_ratio"],
        dropout_rate=hp["dropout_rate"],
        l2_reg=hp["l2_reg"],
    )

    total_steps = EPOCHS * len(train_ds_hp)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=hp["learning_rate"],
        decay_steps=total_steps,
        alpha=1e-6,
    )

    if hp["optimizer"] == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif hp["optimizer"] == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=hp["l2_reg"],
        )
    else:
        opt = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            nesterov=True,
        )

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        train_ds_hp,
        validation_data=val_ds_hp,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    val_acc    = float(max(history.history["val_accuracy"]))
    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    total_ep   = len(history.history["val_accuracy"])
    elapsed    = time.time() - t0

    _, test_acc = model.evaluate(test_ds_hp, verbose=0)

    history_dict = {
        "train_accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy":   [float(x) for x in history.history["val_accuracy"]],
        "train_loss":     [float(x) for x in history.history["loss"]],
        "val_loss":       [float(x) for x in history.history["val_loss"]],
    }

    del model
    tf.keras.backend.clear_session()

    return val_acc, float(test_acc), history_dict, best_epoch, total_ep, elapsed

def evaluate(hyperparams: dict, seed: int, trial_id: int,
             result_cache: dict) -> float:
    h = config_hash(hyperparams)

    if h in result_cache:
        cached   = result_cache[h]
        val_acc  = cached["val_accuracy"]
        test_acc = cached["test_accuracy"]
        source   = cached.get("source", "unknown")
        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(f"    Trial {trial_id:03d} [CACHE:{source}] | "
              f"val={val_acc:.4f} test={test_str} | {hyperparams}")
        save_trial(seed, trial_id, hyperparams,
                   history_dict=None,
                   val_acc=val_acc, test_acc=test_acc,
                   best_epoch=None, total_epochs=None,
                   elapsed=0.0, from_cache=True)
        return val_acc

    val_acc, test_acc, history_dict, best_epoch, total_ep, elapsed = \
        train_and_eval(hyperparams, seed, trial_id)

    save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_ep, elapsed,
               from_cache=False)

    result_cache[h] = {
        "val_accuracy":  val_acc,
        "test_accuracy": test_acc,
        "source":        "pso_summary.csv",
    }

    print(f"    Trial {trial_id:03d} | val={val_acc:.4f} test={test_acc:.4f} "
          f"| ep={total_ep} | {elapsed:.0f}s | {hyperparams}")
    return val_acc

def run_pso_seed(seed: int, result_cache: dict) -> float:
    global _shutdown_requested

    setup_dirs(seed)

    print(f"\n{'='*60}")
    print(f"  PARTICLE SWARM OPTIMISATION — seed {seed}")
    print(f"{'='*60}")

    ckpt, rng_np = load_checkpoint(seed)

    if ckpt is not None:
        generation       = ckpt["generation"]
        positions        = ckpt["positions"]
        velocities       = ckpt["velocities"]
        fitnesses        = ckpt["fitnesses"]
        pbest_positions  = ckpt["pbest_positions"]
        pbest_fitnesses  = ckpt["pbest_fitnesses"]
        evaluated_hashes = ckpt["evaluated_hashes"]
        trial_counter    = ckpt["trial_counter"]
        best_val_acc     = ckpt["best_val_acc"]
        best_hyperparams = ckpt.get("best_hyperparams", None)
        print(f"  Resuming from generation {generation}, "
              f"trial {trial_counter}/{MAX_EVALUATIONS} "
              f"({len(positions)}/{POP_SIZE} particles initialised)")
    else:
        generation       = 0
        positions        = []
        velocities       = []
        fitnesses        = []
        pbest_positions  = []
        pbest_fitnesses  = []
        evaluated_hashes = set()
        trial_counter    = 0
        best_val_acc     = 0.0
        best_hyperparams = None

    def _save(gen):
        save_checkpoint(
            seed, gen,
            positions, velocities, fitnesses,
            pbest_positions, pbest_fitnesses,
            evaluated_hashes,
            trial_counter,
            best_val_acc, best_hyperparams,
            rng_np,
        )

    def _recompute_gbest() -> np.ndarray:
        best_idx = int(np.argmax(pbest_fitnesses))
        return pbest_positions[best_idx].copy()

    if generation == 0:
        print(f"\n  [Gen 0] Initialising swarm ({POP_SIZE} particles)...")
        already_done = len(positions)

        for i in range(already_done, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            pos = random_vector(rng_np)
            vel = rng_np.uniform(-V_MAX / 2, V_MAX / 2)
            hp  = vec_to_dict(pos)
            h   = config_hash(hp)

            val_acc = evaluate(hp, seed, trial_counter, result_cache)

            positions.append(pos)
            velocities.append(vel)
            fitnesses.append(val_acc)
            pbest_positions.append(pos.copy())
            pbest_fitnesses.append(val_acc)
            evaluated_hashes.add(h)
            trial_counter += 1

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen 0): {best_val_acc:.4f} "
                      f"| {best_hyperparams}")

            _save(0)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        generation = 1
        _save(generation)

    for gen in range(generation, N_GENERATIONS + 1):
        if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
            print(f"\n  Budget exhausted or stop requested "
                  f"({trial_counter} trials).")
            break

        gbest = _recompute_gbest()

        print(f"\n  [Gen {gen}] best_so_far={best_val_acc:.4f} "
              f"trials={trial_counter}/{MAX_EVALUATIONS}")

        already_done = trial_counter % POP_SIZE if gen == generation else 0

        if gen > generation or already_done == 0:
            new_positions  = []
            new_velocities = []
            new_fitnesses  = []
        else:
            new_positions  = positions[:already_done]
            new_velocities = velocities[:already_done]
            new_fitnesses  = fitnesses[:already_done]

        for i in range(already_done, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            new_vel = pso_update_velocity(
                velocities[i], positions[i],
                pbest_positions[i], gbest,
                rng_np,
            )
            new_pos = pso_update_position(positions[i], new_vel)
            hp      = vec_to_dict(new_pos)
            h       = config_hash(hp)

            val_acc = evaluate(hp, seed, trial_counter, result_cache)
            evaluated_hashes.add(h)
            trial_counter += 1

            if val_acc > pbest_fitnesses[i]:
                pbest_positions[i] = new_pos.copy()
                pbest_fitnesses[i] = val_acc

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                gbest            = new_pos.copy()
                print(f"    ★ New best (gen {gen}): {best_val_acc:.4f} "
                      f"| {best_hyperparams}")

            new_positions.append(new_pos)
            new_velocities.append(new_vel)
            new_fitnesses.append(val_acc)

            positions  = new_positions  + positions[len(new_positions):]
            velocities = new_velocities + velocities[len(new_velocities):]
            fitnesses  = new_fitnesses  + fitnesses[len(new_fitnesses):]
            _save(gen)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        if len(new_positions) == POP_SIZE:
            positions  = new_positions
            velocities = new_velocities
            fitnesses  = new_fitnesses
            generation = gen + 1
            _save(generation)

    print(f"\n  [DONE] seed={seed} | best_val_acc={best_val_acc:.4f} "
          f"| best_hyperparams={best_hyperparams} "
          f"| total_trials={trial_counter}")
    return best_val_acc

def plot_results():
    import matplotlib.pyplot as plt
    import seaborn as sns

    csv_p = summary_csv_path()
    if not csv_p.exists():
        print("No results to plot yet.")
        return

    df     = pd.read_csv(csv_p)
    df_pso = df[df["method"] == "particle_swarm_optimization"]

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ax    = plt.subplots(figsize=(10, 5))
    all_curves = []

    for seed in SEEDS:
        seed_df = df_pso[df_pso["seed"] == seed].sort_values("trial_id")
        if len(seed_df) == 0:
            continue
        curve = seed_df["val_accuracy"].cummax().values
        all_curves.append(curve)
        ax.plot(range(1, len(curve) + 1), curve,
                alpha=0.3, color="steelblue", linewidth=1)

    if all_curves:
        max_len = max(len(c) for c in all_curves)
        padded  = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge")
            for c in all_curves
        ])
        mean = padded.mean(axis=0)
        std  = padded.std(axis=0)
        x    = np.arange(1, max_len + 1)
        ax.plot(x, mean, color="steelblue", linewidth=2.5,
                label=f"PSO mean (n={len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.2, color="steelblue", label="± std")

    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best validation accuracy (so far)")
    ax.set_title("Particle Swarm Optimisation — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pso_convergence.png", dpi=150)
    plt.close()

    best_per_seed = df_pso.groupby("seed")["val_accuracy"].max().values
    if len(best_per_seed) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([best_per_seed], labels=["Particle Swarm"],
                   patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(
            f"Stability — PSO\n"
            f"mean={best_per_seed.mean():.4f}  std={best_per_seed.std():.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "pso_stability_boxplot.png", dpi=150)
        plt.close()

    hp_df = df_pso[HP_KEYS + ["val_accuracy"]].copy()
    hp_df["optimizer"] = hp_df["optimizer"].map(
        {"adam": 0, "adamw": 1, "sgd": 2}
    )
    hp_df = hp_df.apply(pd.to_numeric, errors="coerce").dropna()

    if len(hp_df) > 5:
        corr = hp_df.corr()[["val_accuracy"]].drop("val_accuracy")
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("HP correlation with val_accuracy")
        plt.tight_layout()
        plt.savefig(plots_dir / "pso_hp_importance.png", dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    cached_mask  = (df_pso["from_cache"] == True
                    if "from_cache" in df_pso.columns
                    else pd.Series([False] * len(df_pso)))
    trained_mask = ~cached_mask
    ax.scatter(df_pso.loc[trained_mask, "trial_id"],
               df_pso.loc[trained_mask, "val_accuracy"],
               alpha=0.4, s=20, color="steelblue", label="trained")
    if cached_mask.any():
        ax.scatter(df_pso.loc[cached_mask, "trial_id"],
                   df_pso.loc[cached_mask, "val_accuracy"],
                   alpha=0.6, s=20, color="darkorange", marker="x",
                   label="cache hit")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("PSO — All trials val_accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pso_all_trials.png", dpi=150)
    plt.close()

    print(f"  Plots saved in {plots_dir}")

def main():
    print("\n" + "=" * 60)
    print("  PSO HPO — EfficientNetB0 — Apple Leaf Disease")
    print(f"  Seeds            : {SEEDS}")
    print(f"  Max evaluations  : {MAX_EVALUATIONS}  per seed")
    print(f"  Swarm size       : {POP_SIZE}")
    print(f"  Generations      : {N_GENERATIONS}")
    print(f"  PSO w / c1 / c2  : {PSO_W} / {PSO_C1} / {PSO_C2}")
    print(f"  V_MAX            : {V_MAX.tolist()}")
    print(f"  Budget (max)     : {MAX_EVALUATIONS} evaluations per seed "
          f"(gen 0: {POP_SIZE}, gens 1-{N_GENERATIONS}: up to "
          f"{N_GENERATIONS * POP_SIZE})")
    if EXTERNAL_SUMMARY_CSVS:
        print(f"  External caches  :")
        for p in EXTERNAL_SUMMARY_CSVS:
            print(f"    • {p}")
    else:
        print(f"  External caches  : none")
    print("=" * 60)

    result_cache = build_cache_from_csv()

    results = {}

    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt is not None and ckpt["trial_counter"] >= MAX_EVALUATIONS:
            print(f"\n  Seed {seed} already complete "
                  f"({ckpt['trial_counter']} trials). Skipping.")
            results[seed] = ckpt["best_val_acc"]
            continue

        best = run_pso_seed(seed, result_cache)
        results[seed] = best

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    accs = list(results.values())
    for seed, acc in results.items():
        print(f"  Seed {seed}: best_val_acc = {acc:.4f}")
    if accs:
        print(f"\n  Mean ± Std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Min / Max  : {np.min(accs):.4f} / {np.max(accs):.4f}")

    print("\n" + "=" * 60)
    print("  BEST HYPERPARAMETERS PER SEED")
    print("=" * 60)
    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt:
            print(f"  Seed {seed}: val={ckpt['best_val_acc']:.4f} "
                  f"| {ckpt['best_hyperparams']}")

    print("\nGenerating plots...")
    plot_results()
    print(f"\nDone. All results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()