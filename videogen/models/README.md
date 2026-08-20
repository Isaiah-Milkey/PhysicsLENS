# models/ — one folder per model family

Adding a model family = adding one package here. Nothing is registered
anywhere: `generate.py` dispatches to `models.<family>` by name, so
`--model hunyuan:...` works the moment `models/hunyuan/` exists.

## Contract

Each family package (`models/<family>/__init__.py`) exports:

```python
MODELS: dict          # checkpoint name -> {repo, modes, defaults...}

def generate(prompt, image=None, images=None, mode=None, model=None,
             out=None, seed=0, steps=..., **overrides) -> Path:
    """Generate one clip, write <out>.mp4 + a .json sidecar with the full
    settings (prompt, seed, checkpoint, resolution...), return the Path."""
```

Common vocabulary (identical across families):

- **Conditioning terms** — the sidecar's `conditioning` field and any
  user-facing mode names use: `text`, `text+image`, `text+video`,
  `text+first+last`, `text+references`. Text is always part of the input;
  the visual media pins the starting state. Family-internal names (Wan's
  t2v/i2v, Cosmos's text2world/image2world) stay inside the family.
- **Duration in seconds** — `generate(..., seconds=5)`; 5 s is the default
  everywhere (`models.DEFAULT_SECONDS`). The family converts to its
  checkpoint's frame count with `models.frames_for(seconds, fps)` (snaps to
  the VAE's 4k+1 constraint). An explicit `num_frames` wins.

Rules:

- **The pipeline stays resident.** `generate()` must be cheap to call
  repeatedly — load on first use, cache at module level (see `wan/_PIPES`).
  That is the whole point of this layout.
- **Mode is inferred from inputs**: no image → t2v, one image → i2v.
  Anything fancier (first+last frame, reference subjects) takes an explicit
  `mode=`.
- **Sidecar JSON is mandatory** — every clip must be reproducible from its
  sidecar alone.
- **Unknown kwargs are the family's problem**: accept `**overrides`, ignore
  `None` values, error on nonsense.

## Families that need a cloned upstream repo

Not everything ships in diffusers. A family that needs vendor code keeps it
*inside its own folder*:

```
models/<family>/
  __init__.py     # the contract above; does its own sys.path / subprocess wiring
  vendor/         # git clone of the upstream repo (gitignored)
  setup.sh        # clones + installs whatever the family needs, idempotent
```

Nothing outside the folder should know or care how the family runs its model.

## Current families

| family | source | modes |
|---|---|---|
| `wan` | diffusers (7 checkpoints, Wan 2.1/2.2) | t2v, i2v, flf, ref |
| `cosmos2.5` (folder `cosmos2_5`) | diffusers (Cosmos-Predict2.5 2B/14B, unified checkpoint) | text, text+image, text+video — inferred from `image=`/`video=` |
| `cosmos3` | diffusers (Cosmos3 omnimodels: nano 33 GB, super-i2v ~120 GB) | text, text+image, text+video — inferred; `super-i2v*` need an image |

Family names with version dots map to folders with underscores
(`--model cosmos2.5` → `models/cosmos2_5/`), so `cosmos3` can sit beside
`cosmos2_5` later.
