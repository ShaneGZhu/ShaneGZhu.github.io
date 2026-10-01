import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

/**
 * The controlled vocabulary for subject tags.
 *
 * Tags describe one dimension only — the subject. What a piece *is* (a tutorial,
 * a reference page) is the separate `kind` field below, because mixing the two in
 * one list makes a tag page unpredictable: clicking "Inference" would give you a
 * subject while clicking "Reference" would give you a page type.
 *
 * Keeping it an enum means a typo or an invented tag fails the build instead of
 * quietly creating a tag page with one post on it. Add a term here when a real
 * need appears, rather than tagging each post with something new.
 */
const TOPICS = [
  'Inference', // serving, scheduling, batching, memory
  'Kernels', // operators, performance work, profiling
  'RL', // reinforcement learning infrastructure
  'Tooling', // frameworks, build and engineering practice
  'Writing', // about this blog itself
] as const;

/** What a piece is, as opposed to what it is about. */
const KINDS = ['Note', 'Tutorial', 'Reference'] as const;

const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      pubDate: z.coerce.date(),
      updatedDate: z.coerce.date().optional(),
      /** Subject tags, from the controlled list. */
      tags: z.array(z.enum(TOPICS)).default([]),
      /** Shown as a badge on the post; `Note` is the default and is not badged. */
      kind: z.enum(KINDS).default('Note'),
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
    /** Subject tags, from the same controlled list as posts. */
    tags: z.array(z.enum(TOPICS)).default([]),
  }),
});

export const collections = { posts, reading };
