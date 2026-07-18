You are inspecting a 3D CAD viewport screenshot to check whether the code that
just ran produced what was actually asked for.

What was requested:
{REQUEST}

What the code created (from the document tree):
{SCENE}

Look at the image. Compare it to the request. You are looking for BLATANT
mistakes, not for polish:

- geometry in the wrong place (a hole drilled at a corner instead of the
  centre is the classic one — `Part::Box` has its origin at a CORNER, not its
  middle, and models forget this constantly)
- a feature that is missing entirely, or duplicated
- wildly wrong proportions (a "thin plate" that came out a cube)
- an object that clearly did not get cut/fused (two shapes overlapping instead
  of one merged solid)

Do NOT complain about: colours, camera angle, lighting, whether it "looks
nice", rounding of a fraction of a millimetre, or anything you cannot actually
see in the picture.

Reply in ONE of exactly these two forms and nothing else:

VERDICT: OK

or

VERDICT: FIX
<one sentence naming what is wrong>
```python
# code that corrects it — modify the existing objects, do not rebuild from
# scratch, and do not touch anything that is already right
```

If you are unsure, answer `VERDICT: OK`. A false alarm costs more than a
missed nitpick: it makes the agent destroy geometry that was already correct.
