# ShaneGZhu.github.io

Personal blog of Shengguang Zhu, built with [Astro](https://astro.build) and
deployed to GitHub Pages. Posts are Markdown or MDX with math,
syntax-highlighted code, callouts, captioned figures and tables — in English or
Chinese.

## Quick start

```bash
npm install
npm run dev      # http://localhost:4321
```

| Command | What it does |
| --- | --- |
| `npm run dev` | Dev server with hot reload; drafts are visible |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the built `dist/` locally |
| `npm run check` | Type-check Astro, TypeScript and content frontmatter |

Node 22.12 or newer is required (Astro 7's engine constraint).

## Writing a post

Create a file under `src/content/posts/`. The filename becomes the URL:
`src/content/posts/my-post.md` → `/posts/my-post`.

```markdown
---
title: 'A title, in English or Chinese'
description: >-
  One or two sentences. Used in listings and link previews, so
  write it for a reader who has not opened the post yet.
pubDate: 2025-03-09
tags: ['Inference', 'Scheduling']
lang: 'en'          # 'en' or 'zh' — sets the lang attribute and date format
---

Body text starts here.
```

Optional frontmatter:

| Field | Type | Effect |
| --- | --- | --- |
| `updatedDate` | date | Shows an "Updated" line on the post |
| `draft` | boolean | Visible in `npm run dev`, excluded from builds and the feed |
| `featured` | boolean | Pinned to the top of the home page |
| `cover` | image path | Hero image, relative to the file (e.g. `../../assets/x.jpg`) |
| `coverAlt` | string | Alt text and caption for the cover |

Frontmatter is validated by the schema in `src/content.config.ts`. A typo fails
the build with the field name, rather than rendering a broken page.

### Syntax beyond standard Markdown

Math, with `$inline$` and `$$display$$`:

```markdown
The score is $s_{ij} = q_i^\top k_j / \sqrt{d}$, normalized over $j$.

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
```

KaTeX renders at build time, so no JavaScript runs in the browser for math.

Callouts, as `:::` container directives — `note`, `tip`, `info`, `warning`,
`danger`, `quote`:

```markdown
:::warning{title="Read this twice"}
The `title` attribute is optional; it defaults to the callout name.
:::
```

Figures — a paragraph containing only an image becomes a `<figure>`, and the
image title becomes the caption:

```markdown
![Alt text for screen readers](../../assets/diagram.png 'Caption under the image')
```

Images under `src/assets/` are resized, converted to WebP and content-hashed at
build time. Images in `public/` are served as-is.

Also available: GFM tables, task lists, footnotes (`[^1]`), `^superscript^`,
`~subscript~`, and raw HTML such as `<details>` when Markdown falls short.

`src/content/posts/rendering-reference.md` exercises all of it; read the rendered
page at `/posts/rendering-reference` to see what each construct looks like.

## Configuration

`src/site.config.mjs` is the one file to edit for anything site-wide: title,
tagline, description, author, role, email, URL, navigation items, social links,
posts per page, and the `FOCUS` list that drives the "What I do" section on the
home page and the About page. `SITE.url` must match the deployed origin —
canonical links and the sitemap are built from it.

Colours, spacing and type scale live at the top of `src/styles/global.css` as
custom properties, with the dark theme as a single `[data-theme='dark']` block.
Article styling is in `src/styles/prose.css`.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which type-checks,
builds, and publishes to GitHub Pages.

One-time repository setup: **Settings → Pages → Build and deployment → Source:
GitHub Actions**. No tokens or deploy keys are needed; the workflow uses the
built-in OIDC token.

### A custom domain

Add a `CNAME` file in `public/` containing the bare domain, point the DNS records
at GitHub, and update `SITE.url` in `src/site.config.mjs` to match.

## Layout

```
src/
├── assets/          images referenced from posts (optimized at build)
├── components/      Header, Footer, PostCard, TableOfContents, Pagination, Icon
├── content/posts/   the posts themselves
├── layouts/         BaseLayout (document shell), PostLayout (articles)
├── markdown/        Markdown pipeline plugins (math, callouts, figures, …)
├── pages/           routes, including 404.astro
├── styles/          global.css (tokens, chrome), prose.css (article body)
├── utils/           post queries, tag grouping, date and reading-time helpers
├── content.config.ts   frontmatter schema
└── site.config.mjs     site metadata and navigation
```

## Notes

- The Markdown pipeline is configured in `astro.config.mjs` via Astro's native
  Sätteri processor. The plugins in `src/markdown/` handle math, callouts,
  figures, CJK line breaks, heading anchors, table wrappers and external-link
  attributes.
- Code blocks use Shiki with two themes compiled in, so they recolour with the
  light/dark toggle without shipping a highlighter to the browser. The
  high-contrast GitHub themes are used because several tokens in the plain ones
  fall below WCAG AA on this site's code background.
- A source line break between two Chinese characters is joined rather than turned
  into a space, so Chinese paragraphs can be hard-wrapped in the editor without
  gaps appearing mid-sentence. Mixed-language breaks keep their space.
- Inter and Newsreader are self-hosted: Astro downloads the Latin subsets at
  build time. Chinese text falls through to the system font, which is both
  faster and better-looking than a webfont CJK subset.
- Icons are inlined single-path SVGs in `src/components/Icon.astro` and inherit
  `currentColor`, so they need no icon font, no extra request, and no per-theme
  asset. The social links on the home page, the About page and the footer are all
  driven by `SOCIAL` in `src/site.config.mjs`, whose `icon` field names a glyph
  there; adding a link means adding both.
- `src/assets/avatar.jpg` is the home page portrait. It is stored at 224px for a
  112px box so 2x displays get a sharp image.
- `SITE.postsOnHome` controls how many posts the home page lists before the
  "Browse all posts" link through to the full index.
