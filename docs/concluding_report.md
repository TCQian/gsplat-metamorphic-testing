# Concluding report: metamorphic testing of gsplat's image renderer

This report is written to be read on its own. It covers what was tested, what was found, why the differences happen, and whether the approach is worth reusing.

## Summary

- **What was tested.** `gsplat` (version 1.5.3) is a GPU library that renders images from a cloud of soft, semi-transparent 3D Gaussians. We applied metamorphic testing to its image-rendering step: render the same scene in two ways that, by a known rule (a metamorphic relation), must give the same image, and check that the two images match. This works without knowing the correct image for any scene.
- **How much.** 100,000 randomly generated scenes of nine types, each checked with five metamorphic relations (defined below) on every camera of the scene (1 to 8), plus deliberately broken versions of three of the relations to check that the tests can catch real bugs. Every finding was traced to its cause (section 3).
- **Result.** No crashes and no wrong images. Three checks matched exactly in every one of the 99,628 non-empty scenes: rendering several cameras at once, giving a Gaussian's shape as a matrix or as size and rotation, and scaling the world by a factor that is exact in binary. The shuffle test differed only in scenes with two Gaussians at nearly the same distance. Scaling by other factors and the packed/unpacked test showed small differences in about 14% and 2% of scenes, almost always at one or a few pixels of one camera's image. Every difference we examined was traced to a tiny rounding difference in the computer's arithmetic, which a hard yes/no decision inside the renderer then turned into a visible difference at that pixel.
- **What it means.** We found no bug in the renderer. We found where and why its output is sensitive to rounding. The test setup reliably catches injected bugs, so it is useful as a regression and fuzzing tool, with the caveats listed in section 5.

## How to read this report: the basic idea

**Metamorphic testing.** When you cannot say what the "correct" image for a random scene is, you can still say how two images must relate. For example, shuffling the order of the Gaussians in the scene must not change the image. If the two images differ, something is wrong. Each such rule is called a *relation*. We used five:

| Name | The two ways of rendering the same scene | Expected result |
| --- | --- | --- |
| MR1 | Using the library's "packed" internal layout vs its normal layout | Same image (tiny rounding allowed) |
| MR2 | Rendering several cameras in one call vs one call per camera | Identical |
| MR3 | Giving each Gaussian's shape as a ready-made matrix vs as size and rotation | Identical |
| MR4 | Enlarging (or shrinking) the whole world, including the camera position, by one common factor `k` | Same image (tiny rounding allowed) |
| MR5 | Shuffling the order of the Gaussians in the input | Identical (unless two Gaussians are at exactly the same distance, see section 4) |

"Identical" means every pixel is exactly the same. "Tiny rounding allowed" means a pixel may differ by at most 0.0001 on a scale where full brightness is 1. Each relation renders every camera of the scene in one call and compares every camera's image.

**Checking the checkers.** A test that always says "pass" is useless. So for MR3, MR4 and MR5 we also ran a version with a deliberately introduced mistake (a wrongly ordered rotation, a camera position left unscaled, colors not shuffled along with the Gaussians). These must be reported as failures. They were, in every scene where the picture was not empty, except for the shuffle mistake in scenes with a single Gaussian, where a shuffle cannot change anything (section 1).

**Outcome labels.** Each test in each scene gets one label:

| Label | Meaning |
| --- | --- |
| `pass` | The two images match within the allowed difference |
| `violation` | They differ by more than allowed, in a way not explained by the cases below |
| `isolated_discontinuity` | They differ at only a handful of pixels (at most 0.05% of any one camera's image) and by at most 0.004, the most that one "just-barely-visible Gaussian" flip can cause (section 3) |
| `tie_ambiguous` | A `violation` in a scene that contains two Gaussians at nearly the same distance from some camera, where the blending order is a coin flip |
| `vacuous_pass` | They match, but every camera's image is empty (nothing was rendered), so the match proves nothing |
| `crash` | The library raised an error |

## 1. What was run and what happened

**The run.** Each scene is random: 1 to 5,000 Gaussians (one scene type uses exactly one), 1 to 8 cameras (4.5 on average), five image sizes, and random Gaussian sizes, opacities and camera distances. Nine scene types push the library toward awkward cases: nearly transparent or fully opaque Gaussians, tiny Gaussians, huge Gaussians, extremely flat "needle" Gaussians, Gaussians spread far around or behind the camera, Gaussians far away, a single Gaussian, and a frame filled with about four layers of semi-transparent Gaussians (`dense_cover`, opacity 0.05 to 0.4, sized from the camera distance and field of view). The 100,000 scenes took 102 minutes on one GPU (60 ms per scene). Nothing crashed. The run was then repeated with a shuffle that moves every Gaussian (Weaknesses, below): MR1 to MR4 gave identical results in every scene, and the MR5 row below is from the repeat.

**Results over 100,000 scenes** (372 scenes produced an empty image from every camera, so their results count as `vacuous_pass`):

| Relation | `pass` | `isolated_discontinuity` | `tie_ambiguous` | `violation` | `vacuous_pass` |
| --- | --- | --- | --- | --- | --- |
| MR1 packed vs normal layout | 97,729 | 1,898 | 0 | 1 | 372 |
| MR2 several cameras at once | 99,628 | 0 | 0 | 0 | 372 |
| MR3 matrix vs size and rotation | 99,628 | 0 | 0 | 0 | 372 |
| MR4 world scaled by factors that are exact in binary (1/128, 1/4, 4, 128) | 99,628 | 0 | 0 | 0 | 372 |
| MR4 world scaled by any factor (also 0.01 and 100) | 85,045 | 14,071 | 504 | 8 | 372 |
| MR5 shuffled Gaussian order | 97,070 | 0 | 2,558 | 0 | 372 |

What these say:

- **MR2, MR3 and exact-factor MR4 are exactly zero everywhere.** Not one pixel differed in any of the 99,628 non-empty scenes, with up to 5,000 Gaussians, 8 cameras rendered together and 256x256 images. Those paths in the library are bit-for-bit consistent.
- **MR1 and general MR4 show small differences in a minority of scenes.** The largest single-pixel difference in an `isolated_discontinuity` was 0.0039. Larger differences appear only in some of the 9 `violation`s (section 3) and in `tie_ambiguous` scenes (up to 0.28 over a whole image; 177 of the 504 above 0.004).
- **The one MR1 `violation` is a small difference spread over too many pixels.** The scene has 1,740 needle-shaped Gaussians and 6 cameras at 256x256, and is almost fully opaque. Five cameras match within tolerance, apart from one pixel just over it (0.00013) in one camera. In the sixth, 65 pixels differ by more than 0.0001, by at most 0.00033: far below the 0.004 an `isolated_discontinuity` allows, but 65 pixels is more than 0.05% of the image (32 pixels), so the label is `violation`. Bisection on that camera keeps one Gaussian, a needle about 165 times longer than wide (condition number 27,000). It is blended in both layouts, but its opacity differs slightly at every pixel along it, because the two layouts compute its 2D shape with different last digits and its thinness magnifies the difference (a "smooth difference", section 3). Removing that one Gaussian leaves no pixel over tolerance in that camera. The scene also has a near-tied pair of Gaussians, but a single kept Gaussian cannot be an order flip. So it is rounding, not a bug in gsplat, and nothing in the library needs fixing; it shows that the pixel-count rule can be exceeded by a single needle. The one pixel in the other camera is a cutoff flip of another Gaussian.
- **MR5 changed the image in 2,558 scenes (2,583 with the first run's random shuffle), all of them scenes with two Gaussians at nearly the same distance from some camera.** In the 69,673 non-empty scenes with no such pair, shuffling never changed the image.
- **More cameras, more findings.** Each camera's image can hit a rounding case on its own, so the share of scenes with a finding grows with the number of cameras (scene types other than `dense_cover`, 88,940 scenes):

  | Cameras | MR4 `isolated_discontinuity` | MR1 `isolated_discontinuity` |
  | --- | --- | --- |
  | 1 | 4.7% | 0.56% |
  | 2 | 8.0% | 1.05% |
  | 4 | 11.6% | 1.69% |
  | 8 | 18.5% | 3.41% |

- **The planted faults were caught** (mistakes we inserted into our own test on purpose, not bugs in `gsplat`; see "Checking the checkers" above) in every scene with a non-empty image, apart from the shuffle fault in single-Gaussian scenes. See "Weaknesses" below.
- **Two known reports from the library's issue tracker were reproduced** by separate small scripts: an assertion failure when the packed layout is combined with a background color (issue #764), and a mix-up in the rotation convention (issue #630, which is a mistake on the caller's side). Both were already known; they were not new findings.
- **Reproducible.** The first 20,000 scenes were also run on their own beforehand, and every verdict was identical.

### Weaknesses of this study

**1. The planted shuffle fault needs a shuffle that moves every Gaussian.** The other two planted faults (rotation order, unscaled camera position) were detected in every non-empty scene. With a random shuffle, the shuffle fault was missed in 11,077 of 99,628 non-empty scenes:
- 11,044 of the 11,077 are single-Gaussian scenes, where a shuffle cannot change anything.
- In the other 33 (0.037% of the non-empty scenes with more than one Gaussian), replaying each scene showed the cause. In 30, every Gaussian visible in the cameras kept its position under the random shuffle (a "fixed point"), so no visible color moved. In 3, the only moved Gaussian that survives culling (the renderer's early step that discards Gaussians that cannot appear in a camera: behind it, outside the near and far limits, or with a footprint entirely off the image) never reaches a pixel, even when rendered alone, although its opacity (0.15 to 0.61) is far above the 1/255 cutoff; its footprint is off the pixel grid or off screen. Every Gaussian that does reach a pixel stayed in place, so these are fixed-point cases too.
- **Fix.** Both the correct shuffle relation and the planted fault now use a shuffle that moves every Gaussian (a derangement: no Gaussian keeps its position). In the repeated 100,000-scene run the fault is detected in every non-empty scene with more than one Gaussian, the 33 included, and the correct relation still changes the image only in scenes with a near-tied pair (no false alarm). So the miss was a property of the chosen shuffle, not of the relation.

**2. A `pass` is weaker than it looks, because most pixels are background.** A pixel is "covered" if the rendered alpha there is above 0, which means at least one Gaussian contributed at least 1/255 of opacity (gsplat skips anything below that); every other pixel is background and shows the plain background color in both renders. Among the 99,628 non-empty scenes (share of covered pixels, averaged over the cameras):

| Share of pixels covered by any Gaussian | Scenes |
| --- | --- |
| under 1% | 11.4% |
| under 5% | 30.9% |
| under 10% | 42.5% |
| under 50% | 74.1% |

The median is 16% coverage, about 9,400 covered pixels over all cameras. 3,341 scenes have fewer than 100 covered pixels and 48 fewer than 10. Fully opaque pixels (alpha above 0.99) are rarer still: the median scene has none, and only 5,709 scenes have more than half their pixels opaque. Background pixels are trivially equal in the two renders, so only the covered pixels can show a difference. The meaning of `pass` therefore grows with the number of covered pixels (recorded per scene as `n_covered_px`), and the 100,000 passes are not equally strong.
- What softens this: the rotation and camera-position faults were still detected in every non-empty scene, so even sparse images are enough to reveal those mistakes. The scene types with large Gaussians (74% median coverage for huge Gaussians) give far stronger passes.
- What does not: a subtle difference that touches few pixels (the kind this study mostly found) is easiest to miss in a sparse image, and a clean sparse scene proves little about the blending of overlapping Gaussians.
- **The dense scene type helps, but does not fill the frame.** `dense_cover` has a median coverage of 45% (quartiles 36% and 55%; none of its 11,060 scenes is under 10%, and 2% are above 80%), against 11% for the other eight types. A simple simulation had predicted 85% to 98%. An experiment on 60 of its scenes showed that culling is not the limit (every Gaussian is kept in at least one camera). Making every Gaussian two or three times larger raises the median coverage from 43% to 59% and 74%. Small scenes (300 Gaussians) respond strongly (70 to 94% at three times the size), so their footprint size limits them. Large scenes (over 2,000 Gaussians) hardly respond (for example 35% to 46%), so for them the region covered by the Gaussians together limits coverage: it is smaller on screen than the simulation assumed, for a reason not found. Because `dense_cover` is one scene type in nine, 42.5% of non-empty scenes are still under 10% coverage (47.8% without it).

## 2. How bisection works

Whenever a test failed by more than the allowed difference, we wanted to know which Gaussians cause the difference at the worst pixel. A scene can have 5,000 Gaussians, so we search for a small set of Gaussians that is enough to reproduce the problem. The method is called *bisection* (a form of "delta debugging"):

1. Pick the camera whose image shows the problem (the largest difference if it is above 0.004, otherwise the most pixels over the allowed difference), and in it the pixel where the two images differ most. Note the size of the difference there.
2. Remove half of the Gaussians from the scene and render both versions again.
3. If the pixel's difference is still at least half of the original, the removed Gaussians were not needed: leave them out. If not, put them back.
4. Repeat with smaller groups of Gaussians, down to one at a time, until nothing more can be removed.
5. The Gaussians left are the "needed" ones: removing any more of them makes the difference at that pixel fall below half. This is all "needed" means here.

At the end the tool prints the remaining Gaussians: how far from the camera each is, how opaque it is at that pixel in each of the two renders, and the final transmittance T at the pixel (1 means nothing blocks the light, 0 means fully blocked).

Variants pick the pixel differently: by the largest change in final transmittance, or by the largest change in the alpha image, because some failures only show up there. The search has a cap of 20,000 renders per scene. The method was first checked on four cases whose answer was known beforehand, and found the exact Gaussians every time.

Limits: it explains only the one pixel it was pointed at. For the nine `violation`s every differing pixel was then explained the same way (section 3); for every other finding, the remaining differing pixels are presumed to share the cause of the worst one.

## 3. What bisection showed, and how it sorts the findings

We ran bisection on every finding (14,071 + 504 + 1,898 + 9), each at the camera that shows it. In each case the Gaussians left over, and a few measured numbers, fall into one of four patterns.

**Pattern A: a "cutoff flip".** One Gaussian remains. The library skips any Gaussian that would add less than 1/255 (about 0.0039) of opacity at a pixel, treating it as invisible. At the worst pixel, this one Gaussian's opacity is right at that limit in both renders (0.003922 against a limit of 0.003922). One render falls just above the limit and counts the Gaussian, the other falls just below and skips it. The light passing through changes by 0.39%, and the pixel changes by at most about 0.0039 (the Gaussian's opacity times its brightness times the light still reaching it).

**Pattern B: a "depth-order flip".** Two Gaussians remain. Gaussians are blended from front to back, and which is in front matters (red in front of blue is not the same as blue in front of red). Here the two Gaussians are at almost exactly the same distance, as close as one float32 step (1 ULP) (about 1.2e-7 relative). In one render they come out in one order, in the other they are tied or swapped. The same two Gaussians are blended in both renders, so the transmittance barely changes (at most 1.3e-5), but the color does. This pattern can be large, because both Gaussians can be quite opaque: the pair alone changes a pixel by up to 0.38, and big Gaussians affect many pixels.

**Pattern C: an "early stop".** More than one Gaussian remains, and the light passing through the pixel at the end is very close to 0.0001 in one render. The library stops blending at a pixel once almost no light is left (0.0001), and the Gaussian that would take it across that line is not blended. In one render the pixel sits just above the line and the Gaussian counts; in the other it sits just below and the pixel stops. The transmittance changes by a large percentage (for example 0.0001 vs 0.0003), but because so little light is left, the color changes by only a small amount, though not always below 0.004.

**Pattern D: a "smooth difference".** One Gaussian remains, but nothing is skipped or reordered. The Gaussian is an extremely thin needle (length about 60 to 175 times its width), and its opacity is slightly different in the two renders at many pixels (for example 0.4448 against 0.4442). The transmittance changes by 0.01% to 0.1%.

### Results of sorting all findings

| Group | Cutoff flip | Depth-order flip | Early stop | Smooth difference |
| --- | --- | --- | --- | --- |
| MR4 `isolated_discontinuity` (14,071) | 13,592 | 270 | 201 | 8 |
| MR4 `tie_ambiguous` (504) | 72 | 413 | 16 | 3 |
| MR1 `isolated_discontinuity` (1,898) | 1,865 | 0 | 28 | 5 |

The patterns are assigned by a rule on the measured values: whether the transmittance changed, how close the kept Gaussians are in depth, whether one is at the 1/255 limit, and the final light at the pixel.

- **Kept sets of several Gaussians.** 36 cutoff flips keep 2 to 13 Gaussians. Exactly one of them is at the 1/255 limit; the rest are far from it and identical in both renders. They matter because they change how much the one flipped Gaussian contributes to the pixel's color.
- **Three scenes did not fit the rule and were sorted by hand.** All three are needles. In one (an MR1 and an MR4 finding), the needle's opacity differs smoothly, and two Gaussians behind it are kept as well because they change how much that difference shows in the color: a smooth difference. In another, the needle's opacity differs smoothly (0.4448 against 0.4442) and the transmittance changes by 0.107%, just over the 0.1% the rule allows a smooth difference: also a smooth difference. In the third, the needle's opacity is 0.003872 in one render and 0.003963 in the other, on either side of the 1/255 limit but each more than 1% from it: a cutoff flip. They are counted that way in the table.
- **The camera matters.** In three quarters of the findings the camera that shows the finding is not the first camera.
- **Every smooth difference involves a needle** (condition number 3,900 to 30,000, against about 1 for a round Gaussian).

### The nine `violation`s

Each was bisected at the camera that shows it. The condition number below is how strongly rounding errors are amplified for the kept Gaussian's conic (the 2D shape matrix used for blending; about the square of its length-to-width ratio, so a round Gaussian has about 1).

| Relation | Scene type | Gaussians | Pixels over tolerance (worst camera) | Kept Gaussians | Condition number | Cause | Pixels over tolerance without the kept Gaussians |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MR1 | needles | 1,740 | 65 (limit 32), D_max 3.3e-4 | 1 | 27,000 | smooth difference | 0 |
| MR4 | needles | 17 | 209, D_max 1.5e-3 | 1 | 20,000 | cutoff flip | 0 |
| MR4 | needles | 11 | 45, D_max 1.3e-4 | 1 | 20,000 | smooth difference | 0 |
| MR4 | needles | 26 | 136, D_max 3.4e-3 | 1 | 13,000 | cutoff flip | 0 |
| MR4 | needles | 12 | 37, D_max 2.0e-4 | 1 | 19,000 | smooth difference | 0 |
| MR4 | needles | 135 | 806, D_max 4.0e-4 | 1 | 26,000 | smooth difference | 1 (D_max 1.5e-4), a cutoff flip of a second Gaussian |
| MR4 | extreme opacities | 108 | 1, D_max 4.7e-3 | 5 | 1.1 to 5.8 | early stop | 0 |
| MR4 | extreme opacities | 120 | 1, D_max 1.5e-2 | 11 | 1.1 to 3.8 | early stop | 0 |
| MR4 | extreme opacities | 71 | 1, D_max 7.2e-2 | 14 | 1.0 to 3.9 | early stop | 0 |

- **Four smooth differences, two cutoff flips, three early stops.** None is a new kind of cause.
- **The six one-Gaussian cases are all needles** (condition numbers 13,000 to 27,000). The three early stops involve round Gaussians (condition numbers close to 1) with opacity 0.999 or 1, where the remaining light at the pixel crosses 0.0001 in one render but not the other. They exceed the 0.004 cutoff bound because the pixel is nearly opaque.
- **Why they became `violation`s.** Most of the one-Gaussian cases fail the pixel-count rule, not the size rule: a needle's long edge lies near a decision line, so one rounding difference moves many pixels at once (209 and 136 pixels for the two cutoff flips). One Gaussian explains the whole group: it alone reproduces every pixel of the group, and removing it leaves nothing over tolerance.
- **Every pixel, every camera.** The table bisects only the worst pixel of the worst camera. Afterwards every over-tolerance pixel of the nine scenes was explained, in every camera and at both inexact factors (26 images, 2,127 pixels, rgb or alpha): bisect the worst pixel not yet explained, count every pixel that the kept Gaussians alone reproduce to at least half as explained, and repeat. One or two rounds per image explain every pixel (1,726 by smooth differences, 397 by cutoff flips, 4 by early stops), and removing the kept Gaussians leaves nothing over tolerance in any of the 26 images. The few over-tolerance pixels in other cameras (seven in five scenes, up to 2.7e-3) are all cutoff flips of one Gaussian, and so is the pixel left over in the last needle scene (a second, nearly round Gaussian).

### What this tells us about the labels

- **`isolated_discontinuity`** was meant to say "a single pixel crossed the 1/255 limit". Bisection confirms that for 97% of the MR4 cases and 98% of the MR1 cases. The rest are early stops, depth-order flips or smooth differences that happened to touch only a few pixels. So the label is mostly right, and the 0.004 size cap (which a cutoff flip cannot exceed) keeps larger problems out of it, but it does not tell you which cause you are looking at.
- **`tie_ambiguous`** was meant to say "this scene contains a near-tied pair, so a difference may just be the blending order flipping". That is what bisection found for 413 of the 504 scenes (82%). For the other 91 the label is a mislabel: the tie in the scene is real, but the failing pixel has a different cause. The label is attached to the whole scene, so it cannot see which pixel is responsible. In 36 of the 91, the bisected image alone would not even make the scene a `violation`: the image that did is the alpha image or the image at the other scaling factor, which the results file does not record, and it was not bisected.
- **`violation`** does not mean "something other than rounding". All 9 are the patterns above, in needle-shaped or nearly opaque scenes, and most arise from the pixel-count rule (0.05% of an image), which a single needle can exceed. A rule that also looks at the condition number or the number of Gaussians involved could separate them from `violation`s that matter; this was not tried.
- **Early stops** exist in both test families (MR1 and MR4), so they are not specific to scaling.

## 4. Root causes: are they all rounding?

Short answer: yes for everything that was examined, with different strengths of evidence. In every case the starting point is that a computer stores numbers with a limited number of digits (about 7 for the "float32" format used by the library), so two ways of computing the same value can end in slightly different last digits. A tiny difference is harmless in most places. It becomes visible only when something depends on a sharp yes/no decision that the difference lands on the wrong side of. The library has three such decisions:

1. Skip a Gaussian if its opacity is below 1/255.
2. Stop blending a pixel once the remaining light is below 0.0001.
3. Blend Gaussians in order of distance, and for exactly equal distances, in input order.

So the explanation has two parts: **rounding creates a tiny difference, and one of the three yes/no rules turns it into a visible one**. Neither alone is enough; this also explains why most scenes are completely clean.

| Pattern | Where the rounding difference comes from | How firmly this is shown |
| --- | --- | --- |
| Cutoff flip (A) | The Gaussian's conic differs in the last digits between the two renders. With a round Gaussian the difference stays tiny; with a thin needle a rounding error of 6e-8 is multiplied by up to the square of its length-to-width ratio (10,000 for a ratio of 100). | Strong. Using the world-scale factor that is exact in binary gives a difference of exactly 0 in every scene. Nudging a Gaussian's size by one step in the last digit changes its conic as much as the real runs do, and by an amount that grows with thinness in the same proportion. Replaying the library's own arithmetic on each render's conic predicts which render counted the Gaussian in all 59 MR1 cases replayed this way. |
| Depth-order flip (B) | Distances pass through a multiplication and a division by `k` (scaling test) or are shuffled (MR5), which leaves nearly equal distances either tied or in the other order. The library breaks exact ties by input order (checked with a dedicated probe), and distances one step apart are always ordered by distance. | Strong for the mechanism (two Gaussians reproduce the difference, with gaps of at most one number step, and the transmittance unchanged). We did not emulate the arithmetic step by step as for pattern A. |
| Early stop (C) | The same conic rounding as in A, now acting on the 0.0001 limit instead of the 1/255 limit. | Confirmed in the two cases replayed in the library's own arithmetic: one render stops the pixel and the other draws the last Gaussian. Other early-stop cases rest on the bisection signature (final light near 0.0001, more than one Gaussian). |
| Smooth difference (D) | The same conic rounding, with no yes/no rule involved: the opacity itself is simply slightly different. | Strong. For the one Gaussian examined in depth, a reference calculation in double precision (about 16 digits) shows the real conic are identical for both renders. The two float32 renders are off from it by 0.0012 and 0.0004, so at least one is wrong by about that much, which is the expected size for a thin Gaussian. |

**Why do MR1's two renders round differently at all?** The packed and normal layouts run the same source code, but the compiler merges multiplications and additions into combined instructions in a different order in each. For the first step of the Gaussian's shape (rotating its size matrix into camera space), we tested all possible merge patterns on specially built scenes and found one pattern reproduces each layout's result exactly. The two layouts merge opposite products in each sum, which changes the last digit of the first intermediate result. Everything after that is identical, but the thinness amplifies this last-digit difference. This was measured with one compiled version of the library on one compiler, so other compiler versions may choose different patterns.

**What is not simply "float32 precision".** Rounding is the cause of the small difference, but not of its visible size: that comes from the hard rules and from how thin the Gaussian is. Also, there is an unavoidable piece that is neither: when two Gaussians sit at exactly the same distance, the output depends on input order by design, and that is not a numerical error at all.

**Does any case point to something other than rounding?** Not among those examined. Every finding was bisected at its worst pixel, and every over-tolerance pixel of the nine `violation`s, in every camera, is explained. What is not individually shown is the rest of the differing pixels in the other findings (presumed to share the worst pixel's cause), and, in 36 `tie_ambiguous` scenes, the image that made them a `violation`.

## 5. Is this metamorphic testing useful?

**Honest answer: useful as a safety net and fuzzing tool, not shown to find a new bug in this version of `gsplat`.** The study found no wrong image and no new crash. What it found is how and where the output depends on rounding.

**What it does well**

- It detects real faults. The three deliberately broken variants were caught in every scene with a non-empty image (the shuffle fault in every one with more than one Gaussian), so a `pass` from the same tests on correct code is meaningful.
- It is cheap: 60 ms per scene with every camera compared (100,000 scenes in 102 minutes). It needs no reference images and no hand-written expected values.
- The exact-match tests (MR2, MR3, MR4 with binary-exact factors) are very strict. They hold bit for bit over 100,000 scenes and every camera, so any future change that breaks one of those is immediately visible.
- It exposed a fact that is not obvious from reading the code: the two code paths (packed vs normal) are not bit-for-bit identical, with about half of the Gaussian shapes differing in the last digits, and the difference is real and measurable.
- It could be rerun on each new release of the library, or after any change to the rendering code, to catch regressions.

**What it does not do**

- It found no incorrect image, so we cannot say it finds bugs at a useful rate in this version.
- It had to be tuned to avoid false alarms. 15,873 of the 100,000 scenes were flagged at some level, and nearly all were harmless rounding. Without the triage described in section 3 (the labels and the bisection), the output would be too noisy to read. The cost of reading the findings is the real cost of using it.
- It covers only part of the library: rendering an image from scratch (the forward pass), with one kind of random scene, one library version, and one type of GPU (the university cluster's NVIDIA cards). It does not cover the backward pass used in training, other camera types, other color options or the library's other rendering modes.
- Most test images are mostly background, so a pass is weaker than it looks (section 1).
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

The progress notes in `docs/progress/` (weeks 7 to 14) hold the full measurements, the commands used and the open caveats of each stage. Weeks 7 to 12 are earlier runs that compared only the first camera, and the root-cause experiments of section 4; week 13 is the run reported here and its follow-up experiments; week 14 closes the items left open in week 13.
