# Leaf_Disease_Classification
cs7643 final project

## Data preparation notes

`scripts/prepare_data.py` groups images by physical leaf using
`leaf_grouping/leaf-map.json` so train/val splits don't leak augmented copies
of the same leaf across sides. That map only covers ~76% of images overall,
and coverage is *not* evenly spread — it's zero for some classes entirely:

- `Corn_(maize)___*` (all 4 classes)
- `Grape___healthy`
- `Squash___Powdery_mildew`
- `Tomato___Target_Spot`
- `Tomato___Tomato_mosaic_virus`

and partial (30-52%) for `Tomato___Late_blight`, `Tomato___Septoria_leaf_spot`,
`Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Tomato___healthy`, and
`Strawberry___Leaf_scorch` — these mix an `RS_*`-named (leaf-tracked) batch
with a `GHLB*`-named (greenhouse) batch that was never leaf-tracked upstream.

This isn't a matching bug: those images fall back to singleton leaf groups,
which is safe (it can only over-split, never merge distinct leaves, so it
can't leak). It does mean that for the classes above, the "leaf-grouped"
split is effectively — or fully — an image-level split. Keep that in mind
when comparing macro-F1 across classes: those classes aren't benefiting from
leak-safe grouping the way most of the others are. `prepare_data.py` prints a
per-class breakdown of any class under 50% coverage so this stays visible on
every run.
