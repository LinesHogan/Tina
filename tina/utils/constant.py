

# problem/question, (solution), answer
RL_POST_TRAIN_DATASET_MAP = {
    # Main datasets
    "curated_deepscaler": "agentica-org/DeepScaleR-Preview-Dataset", # 40.3k
    "curated_still": "RUC-AIBOX/STILL-3-Preview-RL-Data", # 33k
    "curated_open_rs3": "knoveleng/open-rs", # 7k
    "curated_open_rs2": "knoveleng/open-rs", # 7k
    "curated_open_rs1": "knoveleng/open-s1", # 18.6k
    # Extra datasets
    "curated_limr": "GAIR/LIMR", # 1.39k
    "curated_open_r1": "open-r1/OpenR1-Math-220k",  # default split 93.7k
    "curated_thoughts": "bethgelab/CuratedThoughts", # default split 66.1k
    # Ablation
    "curated_limr_large_lr_ablation": "GAIR/LIMR",
    "curated_limr_small_lr_ablation": "GAIR/LIMR",
    "curated_limr_large_rank_ablation": "GAIR/LIMR",
    "curated_limr_medium_rank_ablation": "GAIR/LIMR",
    "curated_limr_small_rank_ablation": "GAIR/LIMR",
    "curated_limr_tiny_rank_ablation": "GAIR/LIMR",
    "curated_open_rs3_drgrpo_ablation": "knoveleng/open-rs",
    # Experiments in diverseGRPO
    "curated_limr_32_rank_lora": "GAIR/LIMR",
    "curated_limr_32_rank_lora_divers": "GAIR/LIMR",
    "curated_limr_32_rank_lora_divers_2": "knoveleng/open-rs",
    "curated_limr_32_rank_full_divers": "GAIR/LIMR",
    "curated_limr_32_rank_full_divers_2": "knoveleng/open-rs",
    "curated_limr_32_rank_full_divers_3": "GAIR/LIMR",
    "curated_limr_full_divers_4(high_lr)": "GAIR/LIMR",
    "curated_limr_full_divers_5(super_hot_begin)": "GAIR/LIMR",
    "curated_limr_full_divers_6(hot_begin_paragraph)": "GAIR/LIMR",
    "curated_limr_full_divers_7(hot_paragraph)": "GAIR/LIMR",
    "curated_limr_full_divers_8(baseline)": "GAIR/LIMR",
    "curated_limr_full_divers_9(hot_lets)": "GAIR/LIMR", # 6771
    "curated_limr_full_divers_10(high_bsz)": "GAIR/LIMR",
    "curated_limr_full_divers_11(low_lr)": "GAIR/LIMR",
    "curated_limr_full_divers_12(superhot_begin_1248)": "GAIR/LIMR",
    "curated_limr_full_divers_13(hot_after_paragraph)": "GAIR/LIMR",
    "curated_limr_full_divers_14(nice_run)": "GAIR/LIMR",
    "curated_limr_full_divers_15(random_run_symmetry)": "GAIR/LIMR",
    "curated_limr_full_divers_16(random_run_neg)": "GAIR/LIMR",
    "curated_limr_full_divers_17(deprecated)": "GAIR/LIMR",
    "curated_limr_full_divers_18(hot_begin_seed1)": "GAIR/LIMR",
    "curated_limr_full_divers_19(hot_begin_seed2)": "GAIR/LIMR",
    "curated_limr_full_divers_20(hot_begin_seed3)": "GAIR/LIMR",
    "curated_limr_full_divers_21(cosine_warmup_temp_schedule)": "GAIR/LIMR",
    "curated_limr_full_divers_22(temp_schedule_lowLR)": "GAIR/LIMR",
    "curated_limr_full_divers_23(temp_schedule_lowerLR)": "GAIR/LIMR",
    "curated_limr_full_divers_24(temp_schedule_lowLR_lessedcay)": "GAIR/LIMR",
    "curated_limr_full_divers_25(temp_schedule_lowerLR_transferVR)": "GAIR/LIMR",
    "curated_limr_full_divers_26(temp_schedule_lowLR_lessedcay_transferVR)": "GAIR/LIMR",
    # experiments for salt
    "salt_exp1": "./training_data/pararel_unsure.json",
    "salt_exp2": "./training_data/pararel_unsure.json",
    "salt_exp3": "./training_data/pararel_unsure.json",
}