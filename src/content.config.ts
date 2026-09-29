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

export const collections = { posts };
