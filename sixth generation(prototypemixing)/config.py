# config.py
import os
import torch

# --- General Settings ---
SEED = 42
BASE_SAVE_DIR = "pipeline_models_dg_zsl_phase_split_v2" # New directory for this revised logic
DEVICE = torch.device("cpu") # Explicitly set to CPU
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Original line

# --- Data Preprocessing ---
SAMPLING_FREQUENCY = 250
SIGNAL_START_TIME = 6.0
SIGNAL_END_TIME = 8.1
AUGMENTATIONS_PER_IMAGE = 1 # OPTIMIZATION: Reduce augmentations slightly if loading is slow

# --- Feature Extractor ---
FEATURE_TYPE = 'cnn' # 'cnn' or 'handcrafted' - Recommend 'handcrafted' for CPU
if FEATURE_TYPE == 'cnn':
    # Recalculate just in case, ensure consistency
    EXPECTED_SIGNAL_LENGTH = int((SIGNAL_END_TIME - SIGNAL_START_TIME) * SAMPLING_FREQUENCY)
    EMBEDDING_DIM = 128 # OPTIMIZATION: Consider smaller embedding dim (e.g., 32 or 64)
    HANDCRAFTED_DIM = None
else: # handcrafted
    EXPECTED_SIGNAL_LENGTH = None
    EMBEDDING_DIM = 64 # OPTIMIZATION: Consider smaller embedding dim (e.g., 32 or 64)
    HANDCRAFTED_DIM = 5

# --- Phase 1: Base Training (Grape + Strawberry) ---
PHASE1_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning", "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"},
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"},
}
PHASE1_EPOCHS = 70 # OPTIMIZATION: Reduced epochs
PHASE1_BATCH_SIZE = 8 # OPTIMIZATION: Reduced batch size
PHASE1_LR = 1e-4
PHASE1_WEIGHT_DECAY = 1e-5
PHASE1_PATIENCE = 8 # OPTIMIZATION: Reduced patience
PHASE1_VALIDATION_SPLIT = 0.2

# --- Phase 2: Domain Generalization Training (Grape + Strawberry) ---
PHASE2_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning", "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"},
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"},
}
PHASE2_EPOCHS = 70 # OPTIMIZATION: Reduced epochs
PHASE2_LR = 4e-5
PHASE2_WEIGHT_DECAY = 1e-6
PHASE2_PATIENCE = 10 # OPTIMIZATION: Reduced patience
PHASE2_VALIDATION_SPLIT = 0.2

# DG Specific Params (Phase 2)
PHASE2_DOMAINS_PER_EPISODE = 2
PHASE2_BATCH_SIZE_PER_DOMAIN = 16 # OPTIMIZATION: Reduced batch size per domain
PHASE2_USE_FEATURE_MIXING = True
PHASE2_MIXING_ALPHA = 0.6
PHASE2_MIXING_LOSS_WEIGHT = 0.93

# --- Stage 2: Maturity Classifier Training (after Phase 2) ---
STAGE2_EPOCHS = 45 # OPTIMIZATION: Reduced epochs
STAGE2_LR = 5e-4
STAGE2_WEIGHT_DECAY = 3e-7
STAGE2_PATIENCE = 8 # OPTIMIZATION: Reduced patience
STAGE2_BATCH_SIZE = 12 # OPTIMIZATION: Reduced batch size

# --- Phase 3: Zero-Shot Inference (Test on Tomato) ---
PHASE3_NEW_FRUIT_NAME = "tomato"
PHASE3_NEW_FRUIT_PATHS = {
     "ripe": r"D:/大二下学期/CDIO/ctomato/tomato_ripe",
     "rotten": r"D:/大二下学期/CDIO/ctomato/tomato_rotten"
}
PHASE3_ZSL_INIT_KNN = 2
PHASE3_ZSL_INIT_MODE = 'weighted_average'

# --- Test Data Paths (New Grape/Strawberry for Evaluation) ---
# Ensure these point to DIFFERENT datasets than used in Phase 1/2 training/validation
TEST_DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripe_test", "rotten": r"D:/大二下学期/CDIO/grape_rotten_test"}, # <-- CHANGE THESE
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe_test", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten_test"}, # <-- CHANGE THESE
}

# --- File Paths ---
# Phase 1 Outputs
P1_SCALER_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_scaler_{FEATURE_TYPE}.joblib')
P1_MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_feature_extractor_{FEATURE_TYPE}_best.pth')
P1_PROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_domain_prototypes_{FEATURE_TYPE}.pth')
P1_HISTORY_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_training_history_{FEATURE_TYPE}.png')
P1_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_validation_tsne_{FEATURE_TYPE}.png')
P1_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase1_validation_confmat_{FEATURE_TYPE}.png')

# Phase 2 Outputs
P2_MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_feature_extractor_{FEATURE_TYPE}_dg_best.pth')
P2_PROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_domain_prototypes_{FEATURE_TYPE}_dg.pth')
P2_HISTORY_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_training_history_{FEATURE_TYPE}_dg.png')
# Validation plots for Phase 2 (Domain separation and combined prediction)
P2_VAL_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_tsne_{FEATURE_TYPE}.png') # tSNE of features
P2_VAL_DOMAIN_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_domain_confmat_{FEATURE_TYPE}.png') # Domain prediction CM
P2_VAL_COMBINED_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase2_validation_combined_confmat_{FEATURE_TYPE}.png') # Combined prediction CM


# Stage 2 Outputs (Maturity Heads) - uses template
S2_MODEL_SAVE_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_maturity_classifier_{{}}_ft_{FEATURE_TYPE}_best.pth') # {} for domain name
S2_HISTORY_PLOT_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_training_history_{{}}_ft_{FEATURE_TYPE}.png') # {} for domain name
S2_CONFMAT_PLOT_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_validation_confmat_{{}}_ft_{FEATURE_TYPE}.png') # {} for domain name (Validation of individual S2 heads)

# Phase 3 Outputs (ZSL)
P3_ZSL_DOMAIN_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_zsl_domain_confmat_{PHASE3_NEW_FRUIT_NAME}.png')
P3_ZSL_DOMAIN_TSNE_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_zsl_domain_tsne_{PHASE3_NEW_FRUIT_NAME}.png')
P3_ZSL_MATURITY_CONFMAT_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'phase3_zsl_maturity_confmat_{PHASE3_NEW_FRUIT_NAME}.png') # Speculative maturity

# Test Script Outputs
TEST_RESULTS_DIR = os.path.join(BASE_SAVE_DIR, "test_results_v2")
# P1 Test plots (Domain only)
P1_TEST_DOMAIN_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p1_test_domain_confmat_{FEATURE_TYPE}.png')
P1_TEST_TSNE_PATH = os.path.join(TEST_RESULTS_DIR, f'p1_test_tsne_{FEATURE_TYPE}.png')
# P2 Test plots (Domain and Combined)
P2_TEST_DOMAIN_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_domain_confmat_{FEATURE_TYPE}.png') # Domain prediction CM
P2_TEST_COMBINED_CONFMAT_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_combined_confmat_{FEATURE_TYPE}.png') # Combined prediction CM
P2_TEST_TSNE_PATH = os.path.join(TEST_RESULTS_DIR, f'p2_test_tsne_{FEATURE_TYPE}.png') # tSNE of features