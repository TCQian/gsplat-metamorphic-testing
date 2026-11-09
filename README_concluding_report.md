# Concluding report: metamorphic testing of gsplat's image renderer

This report is written to be read on its own. It covers what was tested, what was found, why the differences happen, and whether the approach is worth reusing.

## Summary

- **What was tested.** `gsplat` (version 1.5.3) is a GPU library that renders images from a cloud of soft, semi-transparent 3D Gaussians. We applied metamorphic testing to its image-rendering step: render the same scene in two ways that, by a known rule (a metamorphic relation), must give the same image, and check that the two images match. This works without knowing the correct image for any scene.
- **How much.** 100,000 randomly generated scenes, checked with five metamorphic relations (defined below), plus deliberately broken versions of three of them to check the tests can catch real bugs.
- **Result.** No crashes and no wrong images. Two of the five relations (several cameras at once; matrix vs size and rotation) matched exactly in every scene, and so did the scaling test when its factor is exact in binary. The shuffle test differed only in scenes with two Gaussians at nearly the same distance. The scaling test with other factors and the packed/unpacked test showed small differences in a small fraction of scenes (about 5% and about 0.5% of scenes), almost always at one or a few pixels. Every difference we examined was traced to a tiny rounding difference in the computer's arithmetic, which a hard yes/no decision inside the renderer then turned into a visible difference at that pixel.
- **What it means.** We found no bug in the renderer. We found where and why its output is sensitive to rounding. The test setup reliably catches injected bugs, so it is useful as a regression and fuzzing tool, with caveats listed in section 5.

## How to read this report: the basic idea

**Metamorphic testing.** When you cannot say what the "correct" image for a random scene is, you can still say how two images must relate. For example, shuffling the order of the Gaussians in the scene must not change the image. If the two images differ, something is wrong. Each such rule is called a *relation*. We used five:

| Name | The two ways of rendering the same scene | Expected result |
| --- | --- | --- |
| MR1 | Using the library's "packed" internal layout vs its normal layout | Same image (tiny rounding allowed) |
| MR2 | Rendering several cameras in one call vs one call per camera | Identical |
| MR3 | Giving each Gaussian's shape as a ready-made matrix vs as size and rotation | Identical |
| MR4 | Enlarging (or shrinking) the whole world, including the camera position, by one common factor `k` | Same image (tiny rounding allowed) |
| MR5 | Shuffling the order of the Gaussians in the input | Identical (unless two Gaussians are at exactly the same distance, see section 4) |

"Identical" means every pixel is exactly the same. "Tiny rounding allowed" means a pixel may differ by at most 0.0001 on a scale where full brightness is 1.

**Checking the checkers.** A test that always says "pass" is useless. So for MR3, MR4 and MR5 we also ran a version with a deliberately introduced mistake (a wrongly ordered rotation, a camera position left unscaled, colors not shuffled along with the Gaussians). These must be reported as failures. They were, in every scene where the picture was not empty.

**Outcome labels.** Each test in each scene gets one label:

| Label | Meaning |
| --- | --- |
| `pass` | The two images match within the allowed difference |
| `violation` | They differ by more than allowed, in a way not explained by the cases below |
| `isolated_discontinuity` | They differ at only a handful of pixels (at most 0.05% of the image) and by at most 0.004, the most that one "just-barely-visible Gaussian" flip can cause (section 3) |
| `tie_ambiguous` | A `violation` in a scene that contains two Gaussians at nearly the same distance from the camera, where the blending order is a coin flip |
| `vacuous_pass` | They match, but the image is empty (nothing was rendered), so the match proves nothing |
| `crash` | The library raised an error |

## 1. What was run and what happened

Three runs, each larger than the last. The 100,000-scene run includes the 2,000-scene run as its first part.

| Run | Scenes | Purpose |
| --- | --- | --- |
| Preliminary | 160 (8 scene types x 20) | Check the setup and the tolerances |
| Large | 2,000 | Randomised scenes; first full set of findings |
| Extra-large | 100,000 | Statistics, and the findings that were analysed in detail |

Each scene was random: 1 to 5,000 Gaussians (one scene type uses exactly one), 1 to 8 cameras (only MR2 uses more than the first camera, see below), five image sizes, and random Gaussian sizes, opacities and camera distances. Eight scene types push the library toward awkward cases: nearly transparent or fully opaque Gaussians, tiny Gaussians, huge Gaussians, extremely flat "needle" Gaussians, Gaussians spread far around or behind the camera, Gaussians far away, and a single Gaussian. The 100,000 scenes took 43 minutes on one GPU (about 19 ms per scene). Nothing crashed, and no run was cut short.

**Which cameras are used.** MR2 renders every camera of the scene. MR1, MR3, MR4 and MR5, and their planted-fault versions, render only the first camera ("camera 0"). So every image statistic below that mentions camera 0, such as an empty image or the share of covered pixels, describes the image these relations actually compare, and the other cameras are tested only by MR2.

**Results over 100,000 scenes** (1,424 scenes produced an empty image from camera 0, so their results count as `vacuous_pass`; MR2 only counts as vacuous when every camera is empty, which happened in 441 scenes):

| Relation | `pass` | `isolated_discontinuity` | `tie_ambiguous` | `violation` | `vacuous_pass` |
| --- | --- | --- | --- | --- | --- |
| MR1 packed vs normal layout | 98,044 | 532 | 0 | 0 | 1,424 |
| MR2 several cameras at once | 99,559 | 0 | 0 | 0 | 441 |
| MR3 matrix vs size and rotation | 98,576 | 0 | 0 | 0 | 1,424 |
| MR4 world scaled by factors that are exact in binary (1/128, 1/4, 4, 128) | 98,576 | 0 | 0 | 0 | 1,424 |
| MR4 world scaled by any factor (also 0.01 and 100) | 93,794 | 4,643 | 134 | 5 | 1,424 |
| MR5 shuffled Gaussian order | 97,807 | 0 | 769 | 0 | 1,424 |

What these say:

- **MR2, MR3 and exact-factor MR4 are exactly zero everywhere.** Not one pixel differed in any scene, with up to 5,000 Gaussians and 256x256 images (and, for MR2, up to 8 cameras). This is a strong result: those paths in the library are bit-for-bit consistent.
- **MR1 and general MR4 show small differences in a minority of scenes.** The largest single-pixel difference seen in an `isolated_discontinuity` was 0.0039. Only MR4 had a few cases above that (the 5 `violation`s, with differences up to 0.009 at a pixel, and the `tie_ambiguous` scenes, up to 0.15 in the full scene).
- **MR5 changed the image in 769 scenes, all of them scenes with two Gaussians at nearly the same distance.** In the 78,828 scenes with no such pair, shuffling never changed the image.
- **The planted faults were caught** (mistakes we inserted into our own test on purpose, not bugs in `gsplat`; see "Checking the checkers" above) in every scene with a non-empty image, except for the shuffle fault. See "Weaknesses" below.
- **Two known reports from the library's issue tracker were reproduced** by separate small scripts: an assertion failure when the packed layout is combined with a background color (issue #764), and a mix-up in the rotation convention (issue #630, which is a mistake on the caller's side). Both were already known; they were not new findings.

### Weaknesses of this study

**1. The planted shuffle fault was not detected in 12,440 of 98,576 non-empty scenes.** The other two planted faults (rotation order, unscaled camera position) were detected in every non-empty scene.
- 12,363 of the 12,440 are the single-Gaussian scenes, where a shuffle cannot change anything, so no fault can show.
- The other 77 (59 with Gaussians placed far from the camera, 17 with Gaussians spread widely around and behind the camera, 1 with extreme opacities) are unexplained. In 75 of them exactly one Gaussian is visible from camera 0 (and 2 in the other two); among all scenes with one visible Gaussian, the fault was detected in 1,258 and missed in 75 (5.6%). A likely cause is that the random shuffle left that one Gaussian in its original position, so the planted fault could not change its color; the scenes have only 10 to 176 Gaussians, which makes this likely at about the observed rate. This was not checked, because the shuffle and the visible Gaussian are not recorded in the results, so confirming it needs a rerun on the GPU cluster.
- The consequence is limited but real: for these scenes a `pass` from the shuffle test says nothing about the shuffle, though the tests on those scenes could still catch other faults.

**2. A `pass` is weaker than it looks, because most pixels are background.** A pixel is "covered" if the rendered alpha there is above 0, which means at least one Gaussian contributed at least 1/255 of opacity (gsplat skips anything below that); every other pixel is background and shows the plain background color in both renders. Among the 98,576 non-empty scenes (camera 0):

| Share of pixels covered by any Gaussian | Scenes |
| --- | --- |
| under 1% | 12.8% |
| under 5% | 34.6% |
| under 10% | 47.5% |
| under 50% | 75.4% |

The median is 11% coverage, about 2,000 covered pixels. 8,490 scenes have fewer than 100 covered pixels and 549 fewer than 10. Fully opaque pixels (alpha above 0.99) are rarer still: the median scene has none, and only 6,301 scenes have more than half their pixels opaque. Background pixels are trivially equal in the two renders, so only the covered pixels can show a difference. The meaning of `pass` therefore grows with the number of covered pixels, and the 100,000 passes are not equally strong.
- What softens this: the planted faults were still detected in every non-empty scene for the rotation and camera-position faults, so even sparse images are enough to reveal those mistakes. The scene types with large Gaussians (about 75% coverage for the huge-Gaussian type) give far stronger passes.
- What does not: a subtle difference that touches few pixels (the kind this study mostly found) is easiest to miss in a sparse image, and a clean sparse scene proves little about the blending of overlapping Gaussians. An early design problem of this kind was already fixed once: the huge-Gaussian scene type used to produce fully opaque images, where Gaussians behind the front layer could not change any pixel, and its size range was reduced.
- Possible remedy (not done): report the number of covered or overlapping pixels alongside each verdict, and weight or filter passes by it, or add more scene types with dense coverage.

## 2. How bisection works

Whenever a test failed by more than the allowed difference, we wanted to know which Gaussians cause the difference at the worst pixel. A scene can have 5,000 Gaussians, so we search for a small set of Gaussians that is enough to reproduce the problem. The method is called *bisection* (a form of "delta debugging"):

1. Pick the pixel where the two images differ most. Note the size of the difference there.
2. Remove half of the Gaussians from the scene and render both versions again.
3. If the pixel's difference is still at least half of the original, the removed Gaussians were not needed: leave them out. If not, put them back.
4. Repeat with smaller groups of Gaussians, down to one at a time, until nothing more can be removed.
5. The Gaussians left are the "needed" ones: removing any more of them makes the difference at that pixel fall below half. This is all "needed" means here.

At the end the tool prints the remaining Gaussians: how far from the camera each is, how opaque it is at that pixel in each of the two renders, and the final transmittance T at the pixel (1 means nothing blocks the light, 0 means fully blocked).

Variants pick the pixel differently: by the largest change in final transmittance, or by the largest change in the alpha image, because some failures only show up there. The search has a cap of 20,000 renders per scene. The method was first checked on four cases whose answer was known beforehand, and found the exact Gaussians every time.

Limits: it explains only the one pixel it was pointed at. In scenes with many differing pixels, the others are presumed to share the cause, but that was not checked pixel by pixel.

## 3. What bisection showed, and how it sorts the findings

We ran bisection on every one of the 4,643 + 134 + 532 + 5 findings of the 100,000-scene run. In each case the Gaussians left over, and a few measured numbers, fall into one of four patterns.

**Pattern A: a "cutoff flip".** One Gaussian remains. The library skips any Gaussian that would add less than 1/255 (about 0.0039) of opacity at a pixel, treating it as invisible. At the worst pixel, this one Gaussian's opacity is right at that limit in both renders (0.003922 against a limit of 0.003922). One render falls just above the limit and counts the Gaussian, the other falls just below and skips it. The light passing through changes by 0.39%, and the pixel changes by at most about 0.0039 (the Gaussian's opacity times its brightness times the light still reaching it).

**Pattern B: a "depth-order flip".** Two Gaussians remain. Gaussians are blended from front to back, and which is in front matters (red in front of blue is not the same as blue in front of red). Here the two Gaussians are at almost exactly the same distance, as close as one float32 step (1 ULP) (about 1.2e-7 relative). In one render they come out in one order, in the other they are tied or swapped. The same two Gaussians are blended in both renders, so the transmittance barely changes (at most 6e-6), but the color does. This pattern can be large, because both Gaussians can be quite opaque: the pair alone changes a pixel by up to 0.2, and big Gaussians affect many pixels.

**Pattern C: an "early stop".** More than one Gaussian remains, and the light passing through the pixel at the end is very close to 0.0001 in one render. The library stops blending at a pixel once almost no light is left (0.0001), and the Gaussian that would take it across that line is not blended. In one render the pixel sits just above the line and the Gaussian counts; in the other it sits just below and the pixel stops. The transmittance changes by a large percentage (for example 0.00012 vs 0.0005), but because so little light is left, the color changes by only a small amount, though not always below 0.004.

**Pattern D: a "smooth difference".** One Gaussian remains, but nothing is skipped or reordered. The Gaussian is an extremely thin needle (length about 100 times its width), and its opacity is slightly different in the two renders at many pixels (for example 0.306468 against 0.306767). The transmittance changes by 0.02% to 0.03%.

### Results of sorting all findings (100,000-scene run)

| Group | Cutoff flip | Depth-order flip | Early stop | Smooth difference |
| --- | --- | --- | --- | --- |
| MR4 `isolated_discontinuity` (4,643) | 4,507 | 60 | 71 | 5 |
| MR4 `tie_ambiguous` (134) | 8 | 121 | 4 | 1 |
| MR1 `isolated_discontinuity` (532) | 525 | 0 | 7 | 0 |

(Five MR4 cases first looked like a fifth category because bisection kept 2 to 13 Gaussians for them. Checking each kept Gaussian showed exactly one at the 1/255 limit and the rest far from it and identical in both renders, so they are cutoff flips. The extra Gaussians matter because they change how much the one flipped Gaussian contributes to the pixel's color. They are counted as cutoff flips in the table.)

### What this tells us about the labels

- **`isolated_discontinuity`** was meant to say "a single pixel crossed the 1/255 limit". Bisection confirms that for 97% of the MR4 cases and 99% of the MR1 cases. The rest are early stops, depth-order flips or smooth differences that happened to touch only one pixel. So the label is mostly right, and the 0.004 size cap (which a cutoff flip cannot exceed) keeps larger problems out of it, but it does not tell you which cause you are looking at.
- **`tie_ambiguous`** was meant to say "this scene contains a near-tied pair, so a difference may just be the blending order flipping". For 121 of the 134 scenes (90%) that is exactly what bisection found. For the other 13 (8 cutoff flips, 4 early stops, 1 smooth difference) the label is a mislabel: the tie in the scene is real, but the failing pixel has a different cause. The label is attached to the whole scene, so it cannot see which pixel is responsible.
- **Early stops** exist in both test families (MR1 and MR4), so they are not specific to scaling.
- **The five `violation`s in scenes with no near-tied pair** (so nothing to blame on ties) came out as: two early stops with larger differences (0.007 and 0.009, which exceed the cutoff-flip limit of 0.004), two smooth differences from needle-shaped Gaussians (one affecting 50 pixels and one affecting 280), and one that only failed because its transparency image had 47 differing pixels against a limit of 32.

## 4. Root causes: are they all rounding?

Short answer: yes for everything that was examined, with different strengths of evidence. In every case the starting point is that a computer stores numbers with a limited number of digits (about 7 for the "float32" format used by the library), so two ways of computing the same value can end in slightly different last digits. A tiny difference is harmless in most places. It becomes visible only when something depends on a sharp yes/no decision that the difference lands on the wrong side of. The library has three such decisions:

1. Skip a Gaussian if its opacity is below 1/255.
2. Stop blending a pixel once the remaining light is below 0.0001.
3. Blend Gaussians in order of distance, and for exactly equal distances, in input order.

So the explanation has two parts: **rounding creates a tiny difference, and one of the three yes/no rules turns it into a visible one**. Neither alone is enough; this also explains why most scenes are completely clean.

| Pattern | Where the rounding difference comes from | How firmly this is shown |
| --- | --- | --- |
| Cutoff flip (A) | The Gaussian's conic (the 2D shape matrix used for blending) differs in the last digits between the two renders. With a round Gaussian the difference stays tiny; with a thin needle a rounding error of 6e-8 is multiplied by up to the square of its length-to-width ratio (10,000 for a ratio of 100). | Strong. Using the world-scale factor that is exact in binary gives a difference of exactly 0 in every scene. Nudging a Gaussian's size by one step in the last digit changes its conic as much as the real runs do, and by an amount that grows with thinness in the same proportion. Replaying the library's own arithmetic on each render's conic predicts which render counted the Gaussian in 59 of 59 MR1 cases. |
| Depth-order flip (B) | Distances pass through a multiplication and a division by `k` (scaling test) or are shuffled (MR5), which leaves nearly equal distances either tied or in the other order. The library breaks exact ties by input order (checked with a dedicated probe), and distances one step apart are always ordered by distance. | Strong for the mechanism (two Gaussians reproduce the difference, with gaps of at most one number step, and the transmittance unchanged). We did not emulate the arithmetic step by step as for pattern A. |
| Early stop (C) | The same conic rounding as in A, now acting on the 0.0001 limit instead of the 1/255 limit. | Confirmed in the two cases replayed in the library's own arithmetic: one render stops the pixel and the other draws the last Gaussian. Other early-stop cases rest on the bisection signature (final light near 0.0001, more than one Gaussian). |
| Smooth difference (D) | The same conic rounding, with no yes/no rule involved: the opacity itself is simply slightly different. | Strong. For the one Gaussian examined in depth, a reference calculation in double precision (about 16 digits) shows the real conic are identical for both renders. The two float32 renders are off from it by 0.0012 and 0.0004, so at least one is wrong by about that much, which is the expected size for a thin Gaussian. |

**Why do MR1's two renders round differently at all?** The packed and normal layouts run the same source code, but the compiler merges multiplications and additions into combined instructions in a different order in each. For the first step of the Gaussian's shape (rotating its size matrix into camera space), we tested all possible merge patterns on specially built scenes and found one pattern reproduces each layout's result exactly. The two layouts merge opposite products in each sum, which changes the last digit of the first intermediate result. Everything after that is identical, but the thinness amplifies this last-digit difference. This was measured with one compiled version of the library on one compiler, so other compiler versions may choose different patterns.

**What is not simply "float32 precision".** Rounding is the cause of the small difference, but not of its visible size: that comes from the hard rules and from how thin the Gaussian is. Also, there is an unavoidable piece that is neither: when two Gaussians sit at exactly the same distance, the output depends on input order by design, and that is not a numerical error at all.

**Does any case point to something other than rounding?** Not among those examined. The only cases not individually shown at pixel level are the remaining differing pixels in scenes where only the worst pixel was bisected (presumed to share the same cause).

## 5. Is this metamorphic testing useful?

**Honest answer: useful as a safety net and fuzzing tool, not shown to find a new bug in this version of `gsplat`.** The study found no wrong image and no new crash. What it found is how and where the output depends on rounding.

**What it does well**

- It detects real faults. The three deliberately broken variants were caught in every scene with a non-empty image, so a `pass` from the same tests on correct code is meaningful.
- It is cheap: about 19 ms per scene, 100,000 scenes in 43 minutes. It needs no reference images and no hand-written expected values.
- The exact-match tests (MR2, MR3, MR4 with binary-exact factors) are very strict. They hold bit for bit over 100,000 scenes, so any future change that breaks one of those is immediately visible.
- It exposed a fact that is not obvious from reading the code: the two code paths (packed vs normal) are not bit-for-bit identical, with about half of the Gaussian shapes differing in the last digits, and the difference is real and measurable.
- It could be rerun on each new release of the library, or after any change to the rendering code, to catch regressions.

**What it does not do**

- It found no incorrect image, so we cannot say it finds bugs at a useful rate in this version.
- It had to be tuned to avoid false alarms. 5,406 of 100,000 scenes were flagged at some level, and nearly all were harmless rounding. Without the triage described in section 3 (the labels and the bisection), the output would be too noisy to read. The cost of reading the findings is the real cost of using it.
- It covers only part of the library: rendering an image from scratch (the forward pass), with one kind of random scene, one library version, and one type of GPU (the university cluster's NVIDIA cards). It does not cover the backward pass used in training, other camera types, other color options or the library's other rendering modes.
- Most test images are mostly background, so a pass is weaker than it looks.
- Some of the explanations (especially the exact compiler pairing) hold for one compiled build.

**Is it worth using as a fuzzing tool for `gsplat`?** Yes, with a clear purpose: as a regression test and a screen for rendering changes, where an exact match is expected and any new difference is meaningful. It is less suited to hunting for bugs in code that has not changed. If it is used that way, three things would make it more valuable: add relations for the backward pass and other render options; keep the triage labels and the bisection tool so new findings are sorted automatically; and make the near-tied-distance and needle-shaped scene types a larger share of the random mix, since those are where all the findings came from.

## 6. Would double precision inside the kernel help?

*This was not tested. What follows is reasoning from the measurements above, not a result.*

**What would likely improve**

- Double precision stores about 16 digits instead of 7. The last-digit differences between the two code paths would shrink by a factor of hundreds of millions, and the thin-Gaussian amplification (up to 10,000 times, or more for flatter Gaussians) would still leave them far below anything visible.
- The conic would then be close to the true values, not just consistent with each other. For the one thin Gaussian examined in depth, the float32 result was off from the true value by 0.0012; double precision would remove that.
- Cutoff flips, early stops and smooth differences would become much rarer, because the chance that a rounding difference lands on the wrong side of a limit shrinks along with the difference.
- A cheaper version of the idea is to use double precision only for the Gaussian's conic computation, where almost all the amplification happens, and keep the rest in float32. Both layouts would then round the same, very precise, answer back to float32, so they would agree except in extremely rare cases.

**What would not be fixed**

- The hard rules stay. The 1/255 skip, the 0.0001 stop, and ordering by distance are still sharp edges. Any non-zero difference, however small, can land on the wrong side. They would become much rarer, not impossible. It would be a mistake to read a clean 100,000-scene run in double precision as proof that the sharp edges are gone.
- Gaussians at exactly the same distance would still be ordered by input order, since that rule has nothing to do with precision.
- The scaling test with a factor that is not exact in binary (0.01 and 100) would still introduce rounding, just smaller.

**Limitations of doing it**

- **Speed and memory.** Double precision is slower on GPUs. From general knowledge of GPUs (not measured here), it runs at about half the speed of float32 on data-centre cards and can be tens of times slower on consumer cards. Every number takes twice the memory. For a library built for fast training and real-time drawing, this is a heavy cost.
- **The library is written around float32.** The compiled GPU code, the data passed in by callers, the sorting step and the final image all use float32. Changing this would mean touching many places, and the sorting step in particular only keeps 32 bits of each Gaussian's distance. A change in only one step would need conversion at its boundaries.
- **It changes what is being tested.** The tool exists to check the version of `gsplat` that people actually use, which runs in float32. A double-precision build would be a different program. A double-precision *reference* (used here for one Gaussian) is the better use: it tells us how far each float32 result is from the truth without changing the library.
- **Possible mismatch with other parts.** Results would no longer match older float32 results exactly, which could affect people who compare against stored outputs.

**Suggested use.** Rather than changing the library, use a double-precision reference implementation inside the test harness to judge whether a flagged difference is within the expected rounding range. That would let the oracle separate "rounding amplified by a thin Gaussian" from "something genuinely wrong" without relying on the manual triage used in this study.

## Details

The weekly notes `README_week7.md` to `README_week12.md` hold the full measurements, the scripts used and the open caveats for each stage.
