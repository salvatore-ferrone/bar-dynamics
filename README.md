# Bar Dynamics: a pedagogical primer

A learning-in-public project on galactic bar dynamics, building from periodic
orbits up to how bars shape stellar stream morphology. This is a personal
study tool first, and a Jupyter Book website second — the goal is to master
the fundamentals, produce animations/figures worth sharing, and lay the
groundwork for future original research.

Live site: https://ferrone.github.io/bar-dynamics/ *(update the username/org if different)*

## Roadmap

1. Periodic orbits & stability (x1, x2, x3 families)
2. Surfaces of section & frequency analysis (tools)
3. Orbit taxonomy by Jacobi energy band (L3 < Ej < L2 < Ej < L4)
4. Resonances (ILR / CR / OLR / vertical), slow vs. fast bars
5. Chaos & diffusion near separatrices
6. 3D orbits (x1v1/x1v2), buckling & peanut bulges
7. Time-dependent bars: growth + slowdown, resonance sweeping
8. Streams in barred potentials: disk-plane vs. 3D, resonance crossing/capture
9. Cutting edge: self-gravitating streams, Milky Way case studies (e.g. GD-1)

## Repository layout

```
book/
  _config.yml       # Jupyter Book config
  _toc.yml          # table of contents
  intro.md          # landing page
  notebooks/        # one notebook per roadmap topic
  images/           # static figures
  animations/       # gifs/mp4s embedded in notebooks
requirements.txt     # pip deps used by the GitHub Actions build
environment.yml       # conda env for local development
.github/workflows/deploy.yml   # builds & deploys to GitHub Pages
```

## Local development

```bash
conda env create -f environment.yml
conda activate bar-dynamics
jupyter-book build book/
```

Open `book/_build/html/index.html` to preview.

## Publishing

Pushing to `main` triggers the GitHub Actions workflow, which builds the book
and deploys it to GitHub Pages automatically.
