"""Central configuration: paths and modeling constants.

Every script reads paths and hyperparameters from here so the whole
pipeline can be re-run end-to-end without editing individual files.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "ml-1m"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# --- Preprocessing ---------------------------------------------------------
# A rating >= POSITIVE_THRESHOLD counts as an implicit "the user liked this"
# signal. 4 is the conventional cutoff for MovieLens implicit-feedback setups.
POSITIVE_THRESHOLD = 4
# Users need at least this many positives to be split into train/val/test.
MIN_POSITIVES_PER_USER = 5
# Per-user chronological split fractions (train gets the remainder).
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# --- Retrieval (stage 1) ---------------------------------------------------
EMBEDDING_DIM = 64
N_CANDIDATES = 500  # candidates passed from retrieval to the ranker

# --- Evaluation ------------------------------------------------------------
TOP_K = 10
EVAL_KS = (10, 20)

SEED = 42

# --- Artifact filenames ----------------------------------------------------
INTERACTIONS_PATH = ARTIFACTS_DIR / "interactions.parquet"
ITEMS_PATH = ARTIFACTS_DIR / "items.parquet"
USERS_PATH = ARTIFACTS_DIR / "users.parquet"
ITEM_FEATURES_PATH = ARTIFACTS_DIR / "item_features.parquet"
USER_FEATURES_PATH = ARTIFACTS_DIR / "user_features.parquet"
USER_EMBEDDINGS_PATH = ARTIFACTS_DIR / "user_embeddings.npy"
ITEM_EMBEDDINGS_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
RETRIEVAL_CKPT_PATH = ARTIFACTS_DIR / "two_tower.pt"
RANKER_PATH = ARTIFACTS_DIR / "xgb_ranker.json"
RESULTS_PATH = ARTIFACTS_DIR / "eval_results.json"
