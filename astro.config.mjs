// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { satteri } from '@astrojs/markdown-satteri';

import { katexPlugin } from './src/markdown/katex.mjs';
import { calloutPlugin } from './src/markdown/callout.mjs';
import { figurePlugin } from './src/markdown/figure.mjs';
import { cjkLineBreakPlugin } from './src/markdown/cjk.mjs';
import { externalLinksPlugin } from './src/markdown/external-links.mjs';
import { headingAnchorsPlugin, tableWrapperPlugin } from './src/markdown/html-polish.mjs';
import { SITE } from './src/site.config.mjs';

/**
 * Markdown pipeline shared by `.md` and `.mdx`.
 *
 * Sätteri is Astro's native Markdown processor. Feature flags turn on the
 * syntax we want (math, ::: containers, ^sup^ / ~sub~), and the plugins turn
 * those nodes into the HTML the stylesheet expects.
 */
const processor = satteri({
  features: {
    gfm: true,
    math: true,
    directive: true,
    superscript: true,
    subscript: true,
    smartPunctuation: { quotes: false, dashes: true, ellipses: true },
  },
  mdastPlugins: [cjkLineBreakPlugin, calloutPlugin, katexPlugin, figurePlugin],
  hastPlugins: [headingAnchorsPlugin, tableWrapperPlugin, externalLinksPlugin],
});

export default defineConfig({
  site: SITE.url,
  trailingSlash: 'ignore',
  integrations: [mdx(), sitemap()],
  fonts: [
    {
      provider: fontProviders.google(),
      name: 'Inter',
      cssVariable: '--font-inter',
      weights: [400, 500, 600],
      styles: ['normal'],
      subsets: ['latin', 'latin-ext'],
      fallbacks: ['system-ui', 'sans-serif'],
    },
    {
      provider: fontProviders.google(),
      name: 'Newsreader',
      cssVariable: '--font-newsreader',
      weights: [400, 600],
      styles: ['normal', 'italic'],
      subsets: ['latin', 'latin-ext'],
      fallbacks: ['Georgia', 'serif'],
    },
  ],
  markdown: {
    processor,
    syntaxHighlight: { type: 'shiki' },
    shikiConfig: {
      /*
       * Two themes, switched by the `data-theme` attribute on <html>.
       *
       * The high-contrast GitHub variants are used rather than the plain ones:
       * several tokens in `github-light` (notably the orange and green scopes)
       * fall below 4.5:1 against this site's code background, which fails WCAG AA
       * for body-size text.
       */
      themes: { light: 'github-light-high-contrast', dark: 'github-dark-high-contrast' },
      defaultColor: false,
      wrap: false,
    },
  },
  image: {
    // Keeps large photos from blowing up the build output.
    responsiveStyles: true,
    layout: 'constrained',
  },
  build: {
    format: 'directory',
  },
});
