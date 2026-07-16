# Test Videos

Curated clips for exercising the PhysicsLENS pipelines and for demoing
**real vs AI-generated** physics behavior to the team.

## Layout

```
test_videos/
├── real/                       # genuine camera footage — should score LOW suspicion
│   ├── physics_iq/             # Physics-IQ benchmark clips (real lab footage)
│   │   ├── ball-in-basket.mp4         ⭐ matched-pair REAL side
│   │   ├── ball-and-block-fall.mp4
│   │   └── ball-in-sand.mp4
│   └── wikimedia/              # short real clips (bouncing balls, etc.)
└── ai_generated/               # text/image-to-video model output — should score HIGHER
    ├── basketball-onto-crate.mp4      ⭐ matched-pair AI side
    ├── bowling-ball-drop.mp4
    ├── feather-drop.mp4
    └── spring-wan2.6.mp4
```

## ⭐ The matched pair (best for demos)

| | File | Source |
|---|---|---|
| **REAL** | `real/physics_iq/ball-in-basket.mp4` | Physics-IQ clip `0013` (real footage) |
| **AI**   | `ai_generated/basketball-onto-crate.mp4` | Generated from the **first frame of the real clip** |

Same scene, same starting frame — one is real, one is AI. The AI version drops a
basketball into a crate but the ball **disappears on contact (object-permanence
violation)**, which a good VLM suspicion score catches while rating the real clip
low. This side-by-side is the clearest illustration of what the pipeline detects.

## Provenance

- **real/physics_iq/** — clips from the Physics-IQ benchmark (real camera footage).
- **real/wikimedia/** — short public-domain real videos.
- **ai_generated/basketball-onto-crate.mp4** — image-to-video from `ball-in-basket`'s first frame.
- **ai_generated/bowling-ball-drop.mp4**, **feather-drop.mp4** — text-to-video (bowling ball / feather dropped from 3 m).
- **ai_generated/spring-wan2.6.mp4** — Wan 2.6 generative video model.

## Note on git

`*.mp4` is currently gitignored (`.gitignore`), so the clips here are **local only**
and will not be pushed. To share them via the repo, either add a scoped exception
for `test_videos/**` or host the videos externally / via Git LFS.
