# config.py
import os
import torch

# --- General Settings ---
SEED = 42
BASE_SAVE_DIR = "pipeline_models_dg_zsl_phase_split_v2" # Directory for this logic
DEVICE = torch.device("cpu") # Explicitly set to CPU
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Preprocessing ---
SAMPLING_FREQUENCY = 250
SIGNAL_START_TIME = 6.0
SIGNAL_END_TIME = 8.1
AUGMENTATIONS_PER_IMAGE = 1

# --- Feature Extractor ---
FEATURE_TYPE = 'cnn' # 'cnn' or 'handcrafted'
if FEATURE_TYPE == 'cnn':
    EXPECTED_SIGNAL_LENGTH = int((SIGNAL_END_TIME - SIGNAL_START_TIME) * SAMPLING_FREQUENCY)
    EMBEDDING_DIM = 128
    HANDCRAFTED_DIM = None
else: # handcrafted
    EXPECTED_SIGNAL_LENGTH = None
    EMBEDDING_DIM = 64
    HANDCRAFTED_DIM = 5

# --- Phase 1: Base Training (Grape + Strawberry) ---
PHASE1_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning", "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"},
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"},
}
PHASE1_EPOCHS = 90
PHASE1_BATCH_SIZE = 8
PHASE1_LR = 1e-5
PHASE1_WEIGHT_DECAY = 1e-5
PHASE1_PATIENCE = 8
PHASE1_VALIDATION_SPLIT = 0.2
NUM_MATURITY_CLASSES = 2 # Ripe, Rotten

# --- Phase 2: Domain Generalization Training (Grape + Strawberry) ---
PHASE2_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning", "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"},
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"},
}
PHASE2_EPOCHS = 70
PHASE2_LR = 8e-6
PHASE2_WEIGHT_DECAY = 1e-6
PHASE2_PATIENCE = 10
PHASE2_VALIDATION_SPLIT = 0.2

# DG Specific Params (Phase 2)
PHASE2_DOMAINS_PER_EPISODE = 2
PHASE2_BATCH_SIZE_PER_DOMAIN = 16
PHASE2_USE_FEATURE_MIXING = True
PHASE2_MIXING_ALPHA = 0.6
PHASE2_MIXING_LOSS_WEIGHT = 0.9

# --- Stage 2: Maturity Classifier Training (after Phase 2) ---
# NOTE: These are NO LONGER USED in the primary Phase 3 inference path
# with the sub-prototype approach, but kept here for completeness or alternative tests.
STAGE2_EPOCHS = 60
STAGE2_LR = 5e-4
STAGE2_WEIGHT_DECAY = 3e-7
STAGE2_PATIENCE = 8
STAGE2_BATCH_SIZE = 12

# --- Phase 3: Zero-Shot Inference (Test on Tomato) ---
PHASE3_NEW_FRUIT_NAME = "tomato"
PHASE3_NEW_FRUIT_PATHS = {
     "ripe": r"D:/大二下学期/CDIO/tomato_ripe_test",
     "rotten": r"D:/大二下学期/CDIO/tomato_rotten_test"
}

# --- Test Data Paths (For final evaluation of trained models) ---
TEST_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripe_test", "rotten": r"D:/大二下学期/CDIO/grape_rotten_test"}, # <-- Ensure distinct test sets
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe_test", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten_test"}, # <-- Ensure distinct test sets
    "tomato":{"ripe": r"D:/大二下学期/CDIO/tomato_ripe_test","rotten": r"D:/大二下学期/CDIO/tomato_rotten_test"}
}

# --- File Paths ---
# Phase 1 Outputs
P1_SCALER_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_scaler_{FEATURE_TYPE}.joblib')
P1_MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_feature_extractor_{FEATURE_TYPE}_best.pth')
P1_PROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_domain_prototypes_{FEATURE_TYPE}.pth') # Domain prototypes
P1_SUBPROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_subprototypes_{FEATURE_TYPE}.pth') # Maturity Sub-prototypes
P1_HISTORY_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_training_history_{FEATURE_TYPE}.png')
P1_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_validation_tsne_{FEATURE_TYPE}.png')
P1_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_validation_confmat_{FEATURE_TYPE}.png')

# Phase 2 Outputs
P2_MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_feature_extractor_{FEATURE_TYPE}_dg_best.pth')
P2_PROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_domain_prototypes_{FEATURE_TYPE}_dg.pth') # Domain prototypes from P2 DG training data
P2_HISTORY_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_training_history_{FEATURE_TYPE}_dg.png')
P2_VAL_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_tsne_{FEATURE_TYPE}.png')
P2_VAL_DOMAIN_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_domain_confmat_{FEATURE_TYPE}.png')
P2_VAL_COMBINED_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_combined_confmat_{FEATURE_TYPE}.png')


# Stage 2 Outputs (Maturity Heads) - uses template - Likely obsolete now
S2_MODEL_SAVE_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_maturity_classifier_{{}}_ft_{FEATURE_TYPE}_best.pth') # {} for domain name
S2_HISTORY_PLOT_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_training_history_{{}}_ft_{FEATURE_TYPE}.png')
S2_CONFMAT_PLOT_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_validation_confmat_{{}}_ft_{FEATURE_TYPE}.png')

# Phase 3 Outputs (ZSL)
P3_ZSL_DOMAIN_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_zsl_domain_confmat_{PHASE3_NEW_FRUIT_NAME}.png') # Domain prediction CM
P3_ZSL_DOMAIN_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_zsl_domain_tsne_{PHASE3_NEW_FRUIT_NAME}.png') # t-SNE based on domain prediction
P3_MATURITY_SUBPROTO_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_maturity_subproto_confmat_{PHASE3_NEW_FRUIT_NAME}.png') # Maturity prediction CM

# Test Script Outputs
TEST_RESULTS_DIR = os.path.join(BASE_SAVE_DIR, "test_results_v2")
P1_TEST_DOMAIN_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p1_test_domain_confmat_{FEATURE_TYPE}.png')
P1_TEST_TSNE_PATH = os.path.join(TEST_RESULTS_DIR, f'p1_test_tsne_{FEATURE_TYPE}.png')
P2_TEST_DOMAIN_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_domain_confmat_{FEATURE_TYPE}.png')
P2_TEST_COMBINED_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_combined_confmat_{FEATURE_TYPE}.png')
P2_TEST_TSNE_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_tsne_{FEATURE_TYPE}.png')

# --- Optional Plotting Enhancements ---
PLOT_DPI = 300
PLOT_FONT_SIZE_TITLE = 14
PLOT_FONT_SIZE_LABELS = 12
PLOT_FONT_SIZE_TICKS = 10
PLOT_FONT_SIZE_LEGEND = 10
PLOT_MARKER_SIZE = 20
PLOT_CMAP = 'viridis' # Or 'Blues', 'plasma', etc.