# ShaneGZhu.github.io

Personal blog of Shengguang Zhu, built with [Astro](https://astro.build) and
deployed to GitHub Pages.

Every page is served under `/blog/`; the domain root redirects there. Routes come
from the file layout — `src/pages/blog/about.astro` → `/blog/about/` — so adding
a page means adding a file under `src/pages/blog/`. Internal links carry the
prefix explicitly, because Astro's `base` option would prefix generated asset
URLs but not hand-written page links, leaving the two inconsistent. Posts are Markdown or MDX with math,
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
`src/content/posts/my-post.md` → `/blog/posts/my-post/`.

```markdown
---
title: 'A title, in English or Chinese'
description: >-
  One or two sentences. Used in listings and link previews, so
  write it for a reader who has not opened the post yet.
pubDate: 2025-03-09
tags: ['Inference']   # subject tags, from a controlled list
kind: 'Tutorial'      # Note (default) | Tutorial | Reference
lang: 'en'            # 'en' or 'zh' — sets the lang attribute and date format
---

Body text starts here.
```

Optional frontmatter:

| Field | Type | Effect |
| --- | --- | --- |
| `updatedDate` | date | Shows an "Updated" line on the post |
| `draft` | boolean | Visible in `npm run dev`, excluded from production builds |
| `featured` | boolean | Pinned to the top of the home page |
| `cover` | image path | Hero image, relative to the file (e.g. `../../assets/x.jpg`) |
| `coverAlt` | string | Alt text and caption for the cover |

Frontmatter is validated by the schema in `src/content.config.ts`. A typo fails
the build with the field name, rather than rendering a broken page.

### Tags and kinds

Tags carry one dimension only — the **subject** — and come from a fixed list in
`src/content.config.ts`:

`Inference` · `Kernels` · `RL` · `Tooling` · `Writing`

Anything outside that list fails the build, naming the values that are allowed.
Add a term to the list when a real need appears rather than inventing a tag per
post: a tag that appears on a single post makes its own tag page useless.

What a piece *is* goes in the separate `kind` field — `Note` (the default, not
badged), `Tutorial`, or `Reference`. Keeping the two apart means clicking a tag
always means the same thing, instead of mixing subjects with page types. The
reading list uses the same tag list; its entries have no kind.

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

## Collecting other people's writing

The `/blog/reading` page lists articles by other people. It is deliberately a
separate collection from `posts`, so the article listings stay a record of what I
wrote, and so the metadata can describe the source rather than pretending to be a
publication date of mine.

One small YAML file per item under `src/content/reading/`:

```yaml
title: 'The original title, as the author wrote it'
url: 'https://example.com/the-article'
author: 'Their name'          # optional
sourcePublished: 2023-06-20   # optional, when they published it
added: 2025-03-01             # required, when I collected it — the sort order
note: >-
  One or two sentences in my own words on why it is worth reading.
tags: ['Inference']           # optional
```

Entries are links with commentary, not republished text: each one points at the
original and the note is mine. The page sorts by `added`, newest first. An entry
does not get a page of its own — if a note grows into a full piece, it belongs in
`posts` instead.

## Configuration

`src/site.config.mjs` is the one file to edit for anything site-wide: title,
description, author, role, email, URL, navigation items, social links, posts per
page, and the `FOCUS` list that drives the "What I do" section on the home page
and the About page. `SITE.url` must match the deployed origin — canonical links
and the sitemap are built from it.

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
scripts/             make-favicons.mjs — regenerates public/ icons from the avatar
src/
├── assets/          images referenced from posts (optimized at build)
├── components/      Header, Footer, PostCard, TableOfContents, Pagination, Icon
├── content/posts/   the posts themselves
├── content/reading/ other people's writing, as small YAML entries
├── layouts/         BaseLayout (document shell), PostLayout (articles)
├── markdown/        Markdown pipeline plugins (math, callouts, figures, …)
├── pages/           index.astro (root redirect) + 404.astro
│   └── blog/        every real page, served under /blog/
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
- Icons are inlined single-path SVGs in `src/components/Icon.astro`, so they need
  no icon font and no extra request. The `size` prop is in `em`, so icons scale
  with the fluid type scale; it reaches the SVG as a `--icon-size` custom
  property, because a scoped `width` rule would override a width attribute. Each one paints in its brand colour from the
  `--brand-*` tokens in `src/styles/global.css`, keyed by icon name; the dark
  theme substitutes lighter values where a brand colour is too dark to see.
  Pass `brand={false}` for a monochrome icon that should follow the text colour.
  The social links on the home page, the About page and the footer are all driven
  by `SOCIAL` in `src/site.config.mjs`, whose `icon` field names a glyph in that
  component; adding a link means adding both, plus a brand token if it needs one.
- `src/assets/banner.jpg` is the home page banner, a wide 16:9 illustration above
  the intro. It keeps its source ratio at every width; a taller box on phones
  would crop the sides, where the illustration's subject sits.
- `src/assets/avatar.jpg` is the About page portrait and the source for the
  favicons. It is stored at 224px for a 112px box so 2x displays get a sharp
  image. After replacing it, run `npm run favicons` to regenerate
  `public/favicon.ico`, `public/icon-256.png` and `public/apple-touch-icon.png`;
  those outputs are committed, so an ordinary build never runs that step.
- `SITE.postsOnHome` controls how many posts the home page lists before the
  "Browse all posts" link through to the full index.
