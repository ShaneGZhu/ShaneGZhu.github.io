import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      pubDate: z.coerce.date(),
      updatedDate: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      /** `en` or `zh` — drives the `lang` attribute and the date formatting. */
      lang: z.enum(['en', 'zh']).default('en'),
      /** Hidden from listings and feeds, still reachable by direct URL. */
      draft: z.boolean().default(false),
      /** Pinned to the top of the home page. */
      featured: z.boolean().default(false),
      /** Optional hero image, optimized at build time. */
      cover: image().optional(),
      coverAlt: z.string().optional(),
    }),
});

/**
 * Other people's writing, collected rather than written.
 *
 * Kept separate from `posts` so the article listings stay a record of what I
 * wrote, and so the metadata can describe the *source* (`url`, `author`,
 * `sourcePublished`) instead of pretending to be my own publication date.
 *
 * These are links with commentary, not republished text: the entry points at the
 * original and the note is my own sentence about why it is worth reading.
 */
const reading = defineCollection({
  loader: glob({ pattern: '**/*.{yml,yaml}', base: './src/content/reading' }),
  schema: z.object({
    /** The original title, as the author wrote it. */
    title: z.string(),
    /** Canonical URL of the original. */
    url: z.url(),
    author: z.string().optional(),
    /** When the original was published, if it is known. */
    sourcePublished: z.coerce.date().optional(),
    /** When I collected it — the sort order for the page. */
    added: z.coerce.date(),
    /** One or two sentences in my own words. */
    note: z.string(),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = { posts, reading };
