# NPU 8 optimization implementations

Numbered directories contain the source and patches used by the measured
optimizations. The shared directory provides the synthetic HSTU workspace
and runner inputs. Later optimization patches depend on the baseline code
selected for their trial.

Experiments 03 and 22 include the effective A/B activation settings. Experiment
08 has an isolated replay comparison and a full training integration. For
experiment 37, apply the verified lookahead patch to the experiment 26 training
workspace. Experiment 39 provides the complete verified A/B deployment bundle
and its original integrity manifest.

Check `SHA256SUMS` to validate the archived source bytes.
