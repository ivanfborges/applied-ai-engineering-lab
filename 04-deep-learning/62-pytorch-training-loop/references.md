# References

Official PyTorch documentation consulted on 2026-11-19, matched to the local
PyTorch 2.15.0+cpu execution where versioned API pages are available.
Dependency bounds remain in [pyproject.toml](../../pyproject.toml); an API
reference is not a claim of cross-version numerical reproducibility.

- [Dataset, TensorDataset, DataLoader and sampling](https://docs.pytorch.org/docs/2.15/data.html)
  — map-style versus iterable datasets, generator controls and partial batches.
- [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  — the explicit forward/loss/backward/update cycle. Tutorial URLs can evolve.
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html)
  — stable binary logit loss, reductions and target semantics.
- [Autograd mechanics and grad modes](https://docs.pytorch.org/docs/2.15/notes/autograd.html)
  — gradient accumulation, inference restrictions and evaluation-mode distinctions.
- [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
  — independent RNG controls and the limits of seeds and deterministic algorithms.

All observations in this topic use locally generated synthetic data. No public
dataset, pretrained weights or externally downloaded example data is used.

## Visual export references

Official guidance consulted on 2026-11-19 for the separate visual lab:

- [Matplotlib FuncAnimation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html)
  and [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)
  — bounded frame updates and GIF export without an external video encoder.
- [Plotly animation frames](https://plotly.com/python/animations/)
  — explicit recorded states, fixed axes and replay controls. Smooth transitions
  are intentionally disabled for measured training states.
- [Standalone HTML export](https://plotly.com/python/interactive-html-export/)
  — inline runtime bundling for offline use.
- [Plotly configuration](https://plotly.com/python/configuration-options/)
  — explicit mouse-wheel zoom and export controls.
- [Button execution and event hooks](https://plotly.com/python/reference/layout/updatemenus/#layout-updatemenus-items-buttons-items-execute)
  and [slider execution controls](https://plotly.com/python/reference/layout/sliders/#layout-sliders-items-steps-items-execute)
  — application-owned replay calls and scoped handling of interrupted animations.
