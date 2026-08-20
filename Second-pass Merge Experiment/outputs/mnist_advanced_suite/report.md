# Second-pass Merge Experiment Report

## few_shot_recovery

Post-merge few-shot calibration with more stable bias-only and ridge-style frozen-feature calibration.

### few_shot_recovery

| Method | Acc mean | Macro mean | Worst-class mean | Local mean |
|---|---:|---:|---:|---:|
| Centralized | 0.9745 | 0.9743 | 0.9591 | 0.9848 |
| regmean_bias_10shot | 0.9220 | 0.9209 | 0.8717 | 0.9210 |
| regmean_bias_1shot | 0.9221 | 0.9208 | 0.8498 | 0.9195 |
| regmean_bias_5shot | 0.9217 | 0.9207 | 0.8711 | 0.9200 |
| RegMean equal | 0.9215 | 0.9203 | 0.8565 | 0.9209 |
| regmean_lastlayer_tune_10shot | 0.9184 | 0.9175 | 0.8696 | 0.9181 |
| regmean_lastlayer_tune_1shot | 0.9095 | 0.9075 | 0.7993 | 0.9076 |
| regmean_lastlayer_tune_5shot | 0.9151 | 0.9142 | 0.8386 | 0.9140 |
| regmean_ridge_10shot | 0.8777 | 0.8767 | 0.8033 | 0.8730 |
| regmean_ridge_1shot | 0.7243 | 0.7190 | 0.5292 | 0.7244 |
| regmean_ridge_5shot | 0.8557 | 0.8549 | 0.7345 | 0.8503 |
| Simple average | 0.9256 | 0.9247 | 0.8823 | 0.9231 |
| simple_bias_10shot | 0.9235 | 0.9226 | 0.8733 | 0.9225 |
| simple_bias_1shot | 0.9246 | 0.9236 | 0.8711 | 0.9228 |
| simple_bias_5shot | 0.9239 | 0.9231 | 0.8683 | 0.9223 |
| simple_lastlayer_tune_10shot | 0.9179 | 0.9170 | 0.8737 | 0.9202 |
| simple_lastlayer_tune_1shot | 0.9147 | 0.9130 | 0.8240 | 0.9122 |
| simple_lastlayer_tune_5shot | 0.9184 | 0.9176 | 0.8337 | 0.9167 |
| simple_ridge_10shot | 0.8963 | 0.8955 | 0.8110 | 0.8952 |
| simple_ridge_1shot | 0.7293 | 0.7236 | 0.5362 | 0.7303 |
| simple_ridge_5shot | 0.8841 | 0.8832 | 0.8188 | 0.8757 |
