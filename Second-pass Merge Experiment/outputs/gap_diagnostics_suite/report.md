# Second-pass Merge Experiment Report

## calibration_strength

Progressively strengthen post-merge calibration to test whether the remaining gap is more than confidence calibration.

### 10shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| RegMean hidden layers | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + bias-only | 0.8729 | 0.8706 | 0.7029 | 0.8776 |
| RegMean hidden + hidden+head tune | 0.8905 | 0.8894 | 0.8248 | 0.8887 |
| RegMean hidden + last-layer tune | 0.8869 | 0.8852 | 0.7769 | 0.8869 |
| RegMean hidden + temperature | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + vector scaling | 0.8802 | 0.8785 | 0.7623 | 0.8814 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + bias-only | 0.8752 | 0.8724 | 0.6682 | 0.8782 |
| Sample weighted + hidden+head tune | 0.9119 | 0.9109 | 0.8599 | 0.9093 |
| Sample weighted + last-layer tune | 0.9061 | 0.9039 | 0.7590 | 0.9068 |
| Sample weighted + temperature | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + vector scaling | 0.9009 | 0.8991 | 0.7601 | 0.9019 |

### 20shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| RegMean hidden layers | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + bias-only | 0.8722 | 0.8700 | 0.7119 | 0.8764 |
| RegMean hidden + hidden+head tune | 0.8904 | 0.8892 | 0.8176 | 0.8902 |
| RegMean hidden + last-layer tune | 0.8883 | 0.8869 | 0.7937 | 0.8896 |
| RegMean hidden + temperature | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + vector scaling | 0.8837 | 0.8826 | 0.7948 | 0.8826 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + bias-only | 0.8737 | 0.8708 | 0.6603 | 0.8771 |
| Sample weighted + hidden+head tune | 0.9131 | 0.9117 | 0.8386 | 0.9121 |
| Sample weighted + last-layer tune | 0.9091 | 0.9075 | 0.8016 | 0.9100 |
| Sample weighted + temperature | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + vector scaling | 0.9049 | 0.9036 | 0.8128 | 0.9044 |

### 50shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| RegMean hidden layers | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + bias-only | 0.8736 | 0.8714 | 0.7141 | 0.8768 |
| RegMean hidden + hidden+head tune | 0.8912 | 0.8901 | 0.8363 | 0.8923 |
| RegMean hidden + last-layer tune | 0.8915 | 0.8899 | 0.7993 | 0.8911 |
| RegMean hidden + temperature | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + vector scaling | 0.8897 | 0.8883 | 0.8072 | 0.8885 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + bias-only | 0.8747 | 0.8719 | 0.6726 | 0.8781 |
| Sample weighted + hidden+head tune | 0.9133 | 0.9119 | 0.8285 | 0.9129 |
| Sample weighted + last-layer tune | 0.9105 | 0.9088 | 0.8173 | 0.9098 |
| Sample weighted + temperature | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + vector scaling | 0.9068 | 0.9054 | 0.8229 | 0.9051 |

### 5shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| RegMean hidden layers | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + bias-only | 0.8734 | 0.8713 | 0.7209 | 0.8765 |
| RegMean hidden + hidden+head tune | 0.8846 | 0.8836 | 0.8080 | 0.8870 |
| RegMean hidden + last-layer tune | 0.8820 | 0.8812 | 0.8060 | 0.8834 |
| RegMean hidden + temperature | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| RegMean hidden + vector scaling | 0.8794 | 0.8791 | 0.8093 | 0.8775 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + bias-only | 0.8763 | 0.8735 | 0.6704 | 0.8794 |
| Sample weighted + hidden+head tune | 0.9099 | 0.9091 | 0.8584 | 0.9090 |
| Sample weighted + last-layer tune | 0.9060 | 0.9042 | 0.7967 | 0.9019 |
| Sample weighted + temperature | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Sample weighted + vector scaling | 0.9027 | 0.9011 | 0.8363 | 0.8998 |

## layerwise_intervention

Restrict tuning or selective RegMean to specific layers and test which interventions shrink the gap most.

### merge_layer_selection

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| regmean_all_layers | 0.8644 | 0.8619 | 0.6749 | 0.8658 |
| RegMean hidden layers | 0.8760 | 0.8741 | 0.7562 | 0.8764 |
| regmean_last_layer | 0.8757 | 0.8734 | 0.6809 | 0.8753 |
| regmean_middle_last | 0.8660 | 0.8628 | 0.5942 | 0.8732 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| Simple average | 0.7940 | 0.7897 | 0.3994 | 0.8026 |

### tune_layer_selection

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9751 | 0.9751 | 0.9603 | 0.9941 |
| regmean_hidden_full_tune | 0.8912 | 0.8898 | 0.8161 | 0.8906 |
| regmean_hidden_hidden_head_tune | 0.8941 | 0.8925 | 0.8061 | 0.8918 |
| regmean_hidden_last_layer_tune | 0.8864 | 0.8847 | 0.7906 | 0.8870 |
| RegMean hidden layers | 0.8687 | 0.8664 | 0.6962 | 0.8728 |
| regmean_hidden_middle_layer_tune | 0.8924 | 0.8907 | 0.7926 | 0.8914 |
| Sample weighted average | 0.8598 | 0.8566 | 0.6166 | 0.8673 |
| sample_weighted_full_tune | 0.9097 | 0.9080 | 0.8150 | 0.9077 |
| Sample weighted + hidden+head tune | 0.9122 | 0.9106 | 0.8184 | 0.9121 |
| Sample weighted + last-layer tune | 0.9063 | 0.9046 | 0.8217 | 0.9056 |
| sample_weighted_middle_layer_tune | 0.9126 | 0.9110 | 0.8139 | 0.9127 |
