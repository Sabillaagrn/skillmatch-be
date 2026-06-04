"""Offline: convert the .keras model to TensorFlow SavedModel format.

Why: SavedModel is the stable, framework-level serving format. It loads faster
than re-parsing the .keras archive and is the recommended artifact to ship in a
server image. The custom AttentionLayer is baked into the graph, so the server
no longer needs the Python class at load time when using the SavedModel path.

Run:  python -m ml.src.export_savedmodel
Output: ml/artifacts/savedmodel/   (a TF SavedModel directory)

Note: TFLite is intentionally NOT produced here. BiLSTM + custom attention adds
conversion/op-compatibility risk, and the model is small enough that in-process
SavedModel meets real-time latency on a free CPU instance. Revisit only if we
move to edge/serverless.
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .attention import AttentionLayer  # registers the custom layer

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    model = tf.keras.models.load_model(
        ARTIFACTS / "skillmatch_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    out = ARTIFACTS / "savedmodel"
    # Keras 3: export() writes a serving-ready SavedModel with a default signature.
    model.export(str(out))
    print(f"Exported SavedModel -> {out}")


if __name__ == "__main__":
    main()
