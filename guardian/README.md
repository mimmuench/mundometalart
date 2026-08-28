# Guardian — design copy monitoring

Watches the marketplaces for other sellers listing our designs.

## Why shape, and not a reverse image search

Reverse image search answers "who else posted *this photograph*". It is the
right tool when someone lifts our studio shot, and useless the moment a
copycat manufactures the piece and photographs it themselves — which is the
case we actually care about.

Our designs are laser-cut silhouettes, so the design *is* a shape. Guardian
extracts that shape from any photograph and compares shapes, which survives a
different wall, a different camera, a different crop, and a mirror flip.

## What is here (phase 0)

| Module | Job |
| --- | --- |
| `etsy_catalog.py` | Etsy's listings CSV → our catalog, plus the search terms that name each design |
| `imagery.py` | Fetches listing photos, cached, at a sane resolution |
| `fingerprint.py` | Photo → shape outline + perceptual hashes. No pixels are kept |
| `matching.py` | A stranger's photo → the listing of ours it matches, with a verdict |

Not built yet: the marketplace crawlers that feed candidate photos in, the
weekly schedule, and the report. `matching.py` is the part they all plug into.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `image_reuse` | Our photograph, reposted — hashes agree and the shape agrees |
| `design_match` | Our design, their photograph |
| `review` | Close enough to look at, not close enough to claim |
| `clear` | Not ours |

## Where the thresholds come from

`python -m guardian.tests.calibrate` builds designs, photographs each one
several ways, and measures how far apart copies and non-copies actually land.
Contour distance is in pixels on a 128×128 grid; lower means more alike:

| Cutoff | Recall | Precision |
| ---: | ---: | ---: |
| 2.0 px | 80.9% | 100% |
| **2.5 px** | **89.2%** | **100%** |
| 3.0 px | 95.1% | 99.7% |
| 3.5 px | 97.8% | 98.8% |

So 2.5px is `design_match` and 3.5px is the edge of `review`.

Three findings from that sweep are baked into the code, and each one was a
copy the pipeline missed before it was fixed:

- **Area overlap is the wrong measure.** Two unrelated filled silhouettes of
  similar bulk overlap almost perfectly. Comparing outlines instead separated
  the cases; area overlap is now only a shortlisting step.
- **A watermark hid a copy.** A logo stamped in a corner joined the mask and
  stretched the frame everything is normalised against. Recall on watermarked
  photos: 17% → 94% once only the artwork's own blob is kept.
- **A dark wall inverted the mask.** Deciding the artwork by which side owns
  the border, rather than by which side is darker, fixed it: 0% → 94%.

These numbers come from synthetic renders, not photographs. They are honest
about the *relative* difficulty of each case, but the absolute thresholds are
due a recalibration against real copies once we have caught a few.

## Running it

```sh
pip install -r ../requirements.txt

python -m guardian.cli catalog guardian/catalog/etsy_listings.csv   # → data/catalog.json
python -m guardian.cli index                                        # → data/index.json
python -m guardian.cli check some_suspect_photo.jpg                 # → verdict

python -m guardian.tests.test_pipeline   # fast regression tests
python -m guardian.tests.calibrate       # the full sweep (~2 min)
```

Refresh the CSV from Etsy: Shop Manager → Settings → Options → Download Data →
*Currently for Sale Listings*.

`data/` is gitignored — downloaded photos and generated reports never enter
the repository.
