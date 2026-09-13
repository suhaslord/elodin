# DSN held-out TranAD+ result

Status: `PASS_HELD_OUT`

Validated on GitHub Actions run `34782047502` using the public DSN_1k dataset, the frozen public TranAD checkpoint, and the official 799/200 train/test track split.

Evaluation protocol:
- Fit scaling on the official 799-track training split only.
- Split the 200 official test tracks into 100 calibration tracks and 100 untouched final-test tracks.
- Freeze the anomaly threshold on calibration only.
- Do not tune threshold on the final-test half.

Frozen threshold: `0.0011112517677247524`

Calibration (100 tracks, 474,544 points):
- F1: `0.4289987808141783`
- Precision: `0.2859600716743574`
- Recall: `0.8583497941407459`
- TP: `108201`
- FP: `270177`
- TN: `78310`
- FN: `17856`

Untouched final test (100 tracks, 488,288 points):
- F1: `0.4061157096017268`
- Precision: `0.26625984939542435`
- Recall: `0.8554501196468356`
- TP: `103672`
- FP: `285692`
- TN: `81406`
- FN: `17518`

Checkpoint architecture was read from the checkpoint weights: 129 features and feed-forward width 2064; the stored `dim_feedforward=16` is the multiplier used by the original TranAD+ training configuration.

These held-out metrics should not be confused with the upstream published `0.9247305754501705` POT F1, which was produced using test-set evaluation/tuning and is therefore not an untouched held-out result under this protocol.

Actions artifact: `dsn-heldout-results`, artifact ID `10325855544`, SHA-256 `8c76b93fe2a164a6a054fc635e75813ba0827648e0ae2b387afaf9aa42abc01b`.
