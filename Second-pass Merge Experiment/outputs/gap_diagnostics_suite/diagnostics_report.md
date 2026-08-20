# Gap Diagnostics Summary

## Calibration Strength

### 5shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_hidden_layers | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_bias_cal | 0.8734 | 0.8713 | 0.7209 |
| regmean_hidden_layers_hidden_head_tune | 0.8846 | 0.8836 | 0.8080 |
| regmean_hidden_layers_last_layer_tune | 0.8820 | 0.8812 | 0.8060 |
| regmean_hidden_layers_temperature | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_vector_scaling | 0.8794 | 0.8791 | 0.8093 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_bias_cal | 0.8763 | 0.8735 | 0.6704 |
| sample_weighted_hidden_head_tune | 0.9099 | 0.9091 | 0.8584 |
| sample_weighted_last_layer_tune | 0.9060 | 0.9042 | 0.7967 |
| sample_weighted_temperature | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_vector_scaling | 0.9027 | 0.9011 | 0.8363 |

### 10shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_hidden_layers | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_bias_cal | 0.8729 | 0.8706 | 0.7029 |
| regmean_hidden_layers_hidden_head_tune | 0.8905 | 0.8894 | 0.8248 |
| regmean_hidden_layers_last_layer_tune | 0.8869 | 0.8852 | 0.7769 |
| regmean_hidden_layers_temperature | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_vector_scaling | 0.8802 | 0.8785 | 0.7623 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_bias_cal | 0.8752 | 0.8724 | 0.6682 |
| sample_weighted_hidden_head_tune | 0.9119 | 0.9109 | 0.8599 |
| sample_weighted_last_layer_tune | 0.9061 | 0.9039 | 0.7590 |
| sample_weighted_temperature | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_vector_scaling | 0.9009 | 0.8991 | 0.7601 |

### 20shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_hidden_layers | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_bias_cal | 0.8722 | 0.8700 | 0.7119 |
| regmean_hidden_layers_hidden_head_tune | 0.8904 | 0.8892 | 0.8176 |
| regmean_hidden_layers_last_layer_tune | 0.8883 | 0.8869 | 0.7937 |
| regmean_hidden_layers_temperature | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_vector_scaling | 0.8837 | 0.8826 | 0.7948 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_bias_cal | 0.8737 | 0.8708 | 0.6603 |
| sample_weighted_hidden_head_tune | 0.9131 | 0.9117 | 0.8386 |
| sample_weighted_last_layer_tune | 0.9091 | 0.9075 | 0.8016 |
| sample_weighted_temperature | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_vector_scaling | 0.9049 | 0.9036 | 0.8128 |

### 50shot_per_class

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_hidden_layers | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_bias_cal | 0.8736 | 0.8714 | 0.7141 |
| regmean_hidden_layers_hidden_head_tune | 0.8912 | 0.8901 | 0.8363 |
| regmean_hidden_layers_last_layer_tune | 0.8915 | 0.8899 | 0.7993 |
| regmean_hidden_layers_temperature | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_layers_vector_scaling | 0.8897 | 0.8883 | 0.8072 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_bias_cal | 0.8747 | 0.8719 | 0.6726 |
| sample_weighted_hidden_head_tune | 0.9133 | 0.9119 | 0.8285 |
| sample_weighted_last_layer_tune | 0.9105 | 0.9088 | 0.8173 |
| sample_weighted_temperature | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_vector_scaling | 0.9068 | 0.9054 | 0.8229 |

## Layerwise Alignment

| Method | Layer2 CKA | Penultimate CKA | Penultimate class-center cosine |
|---|---:|---:|---:|
| simple_avg | 0.6552 | 0.7446 | 0.8659 |
| sample_weighted_avg | 0.6730 | 0.7632 | 0.8751 |
| sample_weighted_avg_last_layer_tune | 0.6799 | 0.7632 | 0.8751 |
| sample_weighted_avg_hidden_head_tune | 0.6861 | 0.7763 | 0.8894 |
| regmean_hidden_layers | 0.6606 | 0.7363 | 0.8940 |
| regmean_hidden_layers_last_layer_tune | 0.6641 | 0.7363 | 0.8940 |
| regmean_hidden_layers_hidden_head_tune | 0.6715 | 0.7477 | 0.8938 |
| centralized | 1.0000 | 1.0000 | 1.0000 |

## Layerwise Intervention

### merge_layer_selection

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_all_layers | 0.8644 | 0.8619 | 0.6749 |
| regmean_hidden_layers | 0.8760 | 0.8741 | 0.7562 |
| regmean_last_layer | 0.8757 | 0.8734 | 0.6809 |
| regmean_middle_last | 0.8660 | 0.8628 | 0.5942 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| simple_avg | 0.7940 | 0.7897 | 0.3994 |

### tune_layer_selection

| Method | Acc mean | Macro mean | Worst-class mean |
|---|---:|---:|---:|
| centralized | 0.9751 | 0.9751 | 0.9603 |
| regmean_hidden_full_tune | 0.8912 | 0.8898 | 0.8161 |
| regmean_hidden_hidden_head_tune | 0.8941 | 0.8925 | 0.8061 |
| regmean_hidden_last_layer_tune | 0.8864 | 0.8847 | 0.7906 |
| regmean_hidden_layers | 0.8687 | 0.8664 | 0.6962 |
| regmean_hidden_middle_layer_tune | 0.8924 | 0.8907 | 0.7926 |
| sample_weighted_avg | 0.8598 | 0.8566 | 0.6166 |
| sample_weighted_full_tune | 0.9097 | 0.9080 | 0.8150 |
| sample_weighted_hidden_head_tune | 0.9122 | 0.9106 | 0.8184 |
| sample_weighted_last_layer_tune | 0.9063 | 0.9046 | 0.8217 |
| sample_weighted_middle_layer_tune | 0.9126 | 0.9110 | 0.8139 |

## Function-Representation Bridge

| Method | Subset accuracy | Symmetric KL | Penultimate CKA |
|---|---:|---:|---:|
| sample_weighted_avg | 0.8580 | 1.6581 | 0.7714 |
| sample_weighted_avg_last_layer_tune | 0.9130 | 1.2883 | 0.7714 |
| sample_weighted_avg_hidden_head_tune | 0.9220 | 1.0792 | 0.7835 |
| regmean_hidden_layers | 0.8690 | 1.0290 | 0.7447 |
| regmean_hidden_layers_last_layer_tune | 0.8870 | 0.8765 | 0.7447 |
| regmean_hidden_layers_hidden_head_tune | 0.8890 | 0.8365 | 0.7555 |
