# Slipstream 3.1 Website

Static release site for Slipstream 3.1.

## Deploy model

This site is intentionally no-build:

- plain `index.html`
- plain `style.css`
- plain `app.js`

That makes it suitable for:

- Cloudflare Pages
- GitHub Pages
- any static file host

## Local preview

From the repo root:

```bash
cd website
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Cloudflare Pages

Recommended settings:

- Framework preset: `None`
- Build command: leave empty
- Build output directory: `website`

## GitHub Pages

Two straightforward options:

1. Copy `website/` into the root of a dedicated site repo, like `mnemos-web`.
2. Publish `website/` from this repo using a GitHub Pages workflow.

## Content intent

This site is optimized for the Slipstream 3.1 release narrative:

- explain the token tax
- show the `SLIP v3` wire format
- make LangGraph adoption concrete
- make it explicit that training is not required for initial adoption
- link to protocol, package, paper, dataset, and model artifacts
