FloPWD-Benchmark

Overcoming Class Imbalance and Domain Shift in UAV-Based Floating Plastic Segmentation

Official code, trained-model predictions, figures, and evaluation scripts for our paper submitted to the Neural Computing and Applications special issue "Emerging Trends in Smart Waste Monitoring and Environmental Surveillance."

Dataset used: FloPWD 2025 — Dal Lake Floating Plastic Waste Detection Dataset (Mendeley Data, DOI: 10.17632/znxjncgjkc.2)

1. Overview

This repository accompanies our paper benchmarking four lightweight semantic segmentation architectures — MobileNetV2Seg, BiSeNetV2, FastSCNN, and ENet — for detecting floating plastic waste in UAV imagery of Dal Lake, Srinagar. We provide:

A common baseline training protocol applied identically to all four models.
An imbalance- and overfitting-aware pipeline (compound Focal-Tversky + weighted cross-entropy loss, Scale-Stratified PlasticMaskBank copy-paste, soft-label MixUp, EMA weights, multi-scale TTA).
A qualitative cross-environment evaluation on three unseen Kashmir lakes (Manasbal, Wular, Nageen) and non-lake backgrounds, with manually annotated ground-truth masks for the displayed images.
All training/evaluation scripts, trained-model prediction outputs, figure-generation code, and result tables used in the paper, for full reproducibility.

2. Repository structure
FloPWD-Benchmark/
├── README.md
├── code/
│   ├── dataset/
│   │   ├── dal_base.py               # baseline dataset class (shared)
│   │   ├── bisenet_dataset.py        # resolve-stage: PlasticMaskBank + compound loss
│   │   ├── enet_dataset.py
│   │   ├── fastscnn_dataset.py
│   │   └── mobilenet_dataset.py
│   │
│   ├── baseline_training/
│   │   ├── train_bisenet.py
│   │   ├── train_enet.py
│   │   ├── train_fastscnn.py
│   │   └── train_mobilenet.py
│   │
│   ├── resolve_training/
│   │   ├── train_bisenet.py          # rename from truncated "isenet_train"
│   │   ├── train_enet.py             # rename from "net_training"
│   │   ├── train_fastscnn.py         # rename from "astscnn_train"
│   │   └── train_mobilenet.py        # rename from "mobilenet_train"
│   │
│   └── inference/
│       ├── bisenet_base_infer.py     # baseline-checkpoint inference
│       └── bisenet_infer1.py         # resolve-checkpoint inference
│       # NOTE: a single generic inference script covers all four via --model flag
│
├── checkpoints_and_logs/             # "checkpoints & logs" → no & or spaces in repo paths
│   ├── bisenet/
│   │   ├── baseline/                 # was checkpoints_bisenetbaseline
│   │   │   ├── best_model_bisenet.pth
│   │   │   └── logs/
│   │   └── resolved/                 # was checkpoints_bisenetv2_resolve
│   │       ├── best_model_bisenet.pth
│   │       ├── logs/                 # train_loss.csv, train_mIoU.csv, val_*.csv, training_metrics_bisenet.csv, etc.
│   │       └── plots/
│   ├── enet/{baseline,resolved}/...
│   ├── fastscnn/{baseline,resolved}/...
│   └── mobilenet/{baseline,resolved}/...
│
├── predicted_images/                 # "predicted images" → underscore, no space
│   ├── val_baseline/                 # baseline checkpoints on Dal Lake val
│   │   ├── val_bisenet/
│   │   ├── val_enet/
│   │   ├── val_fastscnn/
│   │   └── val_mobilenet/
│   ├── val_resolved/                 # was "val after resolve"
│   │   ├── val_bisenet/
│   │   ├── val_enet/
│   │   ├── val_fastscnn/
│   │   └── val_mobilenet/
│   ├── cross_environment/
│   │   ├── manasbal/                 
│   │   │   ├── original/
│   │   │   ├── bisenet_pred/
│   │   │   ├── enet_pred/
│   │   │   ├── fastscnn_pred/
│   │   │   └── mobilenet_pred/
│   │   ├── wuler/                    
│   │   │   └── {original,bisenet_pred,enet_pred,fastscnn_pred,mobilenet_pred}/
│   │   ├── nigeen/                  
│   │   │   └── {original,bisenet_pred,enet_pred,fastscnn_pred,mobilenet_pred}/
│   │   └── non_lake/
│   │       └── {original,bisenet_pred,enet_pred,fastscnn_pred,mobilenet_pred}/
│   └── inference/                    # ad-hoc / demo inference outputs
│
├── figures/           
    ├── fig1_train_val_loss_miou_baseline.png
    ├── fig1_train_val_loss_miou_resolved.png
    ├── fig3_metrics_heatmap_baseline.png
    ├── fig3_metrics_heatmap_resolved.png
    ├── maskbank_diagram.png
    ├── qualitative_comparison_grid_{baseline,resolved}.png
    └── qualitative_{manasbal,wular,nageen,nonlake}_grid.png
   
## Results

### Baseline vs. resolved pipeline (validation set)

| Model | Val mIoU (baseline) | Val mIoU (resolved) | Val recall (baseline → resolved) | Train–val mIoU gap (baseline → resolved) |
|---|---|---|---|---|
| BiSeNetV2 | 0.837 | 0.843 | 0.797 → 0.859 | 13.9 → 3.8 |
| ENet | 0.820 | 0.815 | 0.780 → 0.815 | 10.9 → 4.4 |
| FastSCNN | 0.824 | 0.826 | 0.788 → 0.837 | 11.9 → 1.9 |
| MobileNetV2Seg | 0.834 | 0.839 | 0.796 → 0.845 | 11.5 → 2.8 |

Full per-metric tables and training curves are in `results/*/logs/` and
`figures/`.

### Cross-environment evaluation

Zero-shot predictions (no retraining) on unseen lakes and non-lake
scenes are in `results/cross_environment/`, with overlay figures in
`figures/qualitative_{manasbal,wular,nageen,nonlake}_grid.pdf`. This
evaluation is qualitative — ground-truth masks were annotated only for
the displayed subset and are shown as a visual reference, not used to
compute metrics.



## Citation

If you use this code or the FloPWD 2025 dataset, please cite:

@article{ulhaq2026flopwd,
  title   = {Overcoming Class Imbalance and Domain Shift in UAV-Based Floating Plastic Segmentation},
  author  = {Ul-Haq, Ahsan and Veningston, K.},
  journal = {Neural Computing and Applications},
  year    = {2026},
  note    = {Special Issue: Emerging Trends in Smart Waste Monitoring and Environmental Surveillance}
}

## License

- **Code** in this repository is released under the [MIT License](LICENSE).
- **FloPWD 2025 dataset** is released separately on Mendeley Data under
  its own license — see the dataset page for terms.

---

## Acknowledgements

Department of Computer Science and Engineering, National Institute of
Technology Srinagar. We thank JKSTSC.

---

## Contact

Ahsan Ul-Haq — `ahsan_2025phacse002@nitsri.ac.in`
Veningston K. — `veningstonk@nitsri.ac.in`
