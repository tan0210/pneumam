# Pneumam docs

The documentation site, built with [Astro Starlight](https://starlight.astro.build).

```
bun install
bun run dev      # http://localhost:4321, reloads on save
bun run build    # static site in dist/
```

- Pages are Markdown in `src/content/docs/<section>/`. The folder is the sidebar group; `sidebar.order` in the front matter sets the order.
- Notes use Starlight's asides: `:::note`, `:::tip`, `:::caution`.
- The look (colours, fonts, the ruled band) is in `src/styles/pneumam.css`; code colours are in `ec.config.mjs`.
- `astro.config.mjs` carries a small fix for a Windows build bug in Expressive Code 0.44.3; remove it once that is fixed upstream.
