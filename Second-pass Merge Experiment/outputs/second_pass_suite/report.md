# Second-pass Merge Experiment Report

## alpha_sweep

Fixed local training budget, sweep non-IID strength through Dirichlet alpha.

### alpha_0.1

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9869 |
| Epoch weighted | 0.5620 | 0.5550 | 0.0000 | 0.6440 |
| RegMean epoch weighted | 0.7716 | 0.7658 | 0.3450 | 0.8191 |
| RegMean equal | 0.7716 | 0.7658 | 0.3450 | 0.8191 |
| Simple average | 0.5620 | 0.5550 | 0.0000 | 0.6440 |
| Best single | 0.5304 | 0.5226 | 0.0000 | - |

RegMean epoch weighted minus simple average accuracy: `+0.2096`

### alpha_0.5

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9848 |
| Epoch weighted | 0.8168 | 0.8124 | 0.5157 | 0.8216 |
| RegMean epoch weighted | 0.8813 | 0.8787 | 0.7007 | 0.8796 |
| RegMean equal | 0.8813 | 0.8787 | 0.7007 | 0.8796 |
| Simple average | 0.8168 | 0.8124 | 0.5157 | 0.8216 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0645`

### alpha_1.0

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9847 |
| Epoch weighted | 0.8816 | 0.8809 | 0.7374 | 0.8782 |
| RegMean epoch weighted | 0.9001 | 0.8988 | 0.8307 | 0.8974 |
| RegMean equal | 0.9001 | 0.8988 | 0.8307 | 0.8974 |
| Simple average | 0.8816 | 0.8809 | 0.7374 | 0.8782 |
| Best single | 0.9095 | 0.9090 | 0.7975 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0185`

### alpha_5.0

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9848 |
| Epoch weighted | 0.9211 | 0.9200 | 0.8624 | 0.9184 |
| RegMean epoch weighted | 0.9182 | 0.9169 | 0.8498 | 0.9155 |
| RegMean equal | 0.9182 | 0.9169 | 0.8498 | 0.9155 |
| Simple average | 0.9211 | 0.9200 | 0.8624 | 0.9184 |
| Best single | 0.9238 | 0.9229 | 0.8767 | - |

RegMean epoch weighted minus simple average accuracy: `-0.0029`

## calibration_sweep

Fixed a representative non-IID setting, sweep calibration set size for RegMean statistics.

### calib_128

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.7856 | 0.7798 | 0.3778 | 0.7964 |
| RegMean equal | 0.8123 | 0.8081 | 0.5280 | 0.8162 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `-0.0084`

### calib_16

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.6327 | 0.6260 | 0.3083 | 0.6390 |
| RegMean equal | 0.5949 | 0.5882 | 0.2870 | 0.5970 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `-0.1613`

### calib_256

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.8230 | 0.8173 | 0.3700 | 0.8292 |
| RegMean equal | 0.8438 | 0.8400 | 0.5381 | 0.8563 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0290`

### calib_32

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.6795 | 0.6739 | 0.3890 | 0.6770 |
| RegMean equal | 0.6254 | 0.6194 | 0.3610 | 0.6202 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `-0.1145`

### calib_512

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.8368 | 0.8316 | 0.4092 | 0.8450 |
| RegMean equal | 0.8660 | 0.8626 | 0.5751 | 0.8750 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0428`

### calib_64

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.7370 | 0.7324 | 0.4596 | 0.7417 |
| RegMean equal | 0.7193 | 0.7150 | 0.5067 | 0.7246 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `-0.0570`

## per_class_recovery

Measure which classes are recovered by RegMean relative to simple average.

### dirichlet_medium

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9848 |
| Epoch weighted | 0.8168 | 0.8124 | 0.5157 | 0.8216 |
| RegMean epoch weighted | 0.8813 | 0.8787 | 0.7007 | 0.8796 |
| RegMean equal | 0.8813 | 0.8787 | 0.7007 | 0.8796 |
| Simple average | 0.8168 | 0.8124 | 0.5157 | 0.8216 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0645`

## per_client_recovery

Measure which client-local distributions are recovered by RegMean relative to simple average.

### dirichlet_budget_hetero

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| Epoch weighted | 0.6055 | 0.5994 | 0.0173 | 0.6173 |
| RegMean epoch weighted | 0.8230 | 0.8173 | 0.3700 | 0.8292 |
| RegMean equal | 0.8438 | 0.8400 | 0.5381 | 0.8563 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |
| Best single | 0.9120 | 0.9105 | 0.8171 | - |

RegMean epoch weighted minus simple average accuracy: `+0.0290`
