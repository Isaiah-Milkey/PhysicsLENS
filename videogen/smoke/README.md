# smoke/ — one validation clip per (family, checkpoint, conditioning)

Layout: `smoke/<family>/<checkpoint>_<conditioning>.mp4`, all at the family's
default settings and the common 5-second length. Every clip has a `.json`
sidecar with the full generation settings, and a `*_strip.png` showing
first / middle / last frame for quick visual checks.

- `inputs/` — shared conditioning media. `marble_midair.png` is a frame of a
  glass marble in mid-air above a wooden floor; every `*_text_image` clip
  continues from it with the prompt "the glass marble falls and bounces on
  the wooden floor".
- Every `*_text` clip uses the prompt "a glass marble rolls off a wooden
  table and falls to the floor, static camera, side view".

Same prompts everywhere, so families and checkpoints are directly comparable.
