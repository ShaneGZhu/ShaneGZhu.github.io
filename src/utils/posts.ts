import { getCollection, type CollectionEntry } from 'astro:content';
import Slugger from 'github-slugger';

export type Post = CollectionEntry<'posts'>;

/**
 * Every publishable post, newest first.
 *
 * Drafts are visible while running `astro dev` so they can be previewed, and
 * excluded from production builds, listings and the feed.
 */
export async function getPublishedPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/** Featured posts first, then the rest — used on the home page. */
export function sortFeaturedFirst(posts: Post[]): Post[] {
  return [...posts].sort((a, b) => {
    if (a.data.featured !== b.data.featured) return a.data.featured ? -1 : 1;
    return b.data.pubDate.valueOf() - a.data.pubDate.valueOf();
  });
}

/**
 * URL-safe form of a tag. Deterministic: a fresh slugger per call, so the same
 * label always yields the same slug no matter the call order.
 */
export function tagSlug(label: string): string {
  return new Slugger().slug(label.trim());
}

export interface TagSummary {
  /** Used in `/tags/<slug>`. */
  slug: string;
  /** Original casing, as written in frontmatter. */
  label: string;
  count: number;
}

/**
 * Tag index, most-used first then alphabetical.
 *
 * Tags are matched by slug, so `RDMA`, `rdma` and `RDMA ` are one tag; the first
 * spelling encountered becomes the display label.
 */
export function collectTags(posts: Post[]): TagSummary[] {
  const bySlug = new Map<string, TagSummary>();

  for (const post of posts) {
    for (const raw of post.data.tags) {
      const label = raw.trim();
      if (!label) continue;

      const slug = tagSlug(label);
      if (!slug) continue;

      const existing = bySlug.get(slug);
      if (existing) existing.count += 1;
      else bySlug.set(slug, { slug, label, count: 1 });
    }
  }

  return [...bySlug.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

/** Posts carrying a given tag slug. */
export function postsWithTag(posts: Post[], slug: string): Post[] {
  return posts.filter((post) => post.data.tags.some((tag) => tagSlug(tag) === slug));
}

/** Group posts by publication year, newest year first. */
export function groupByYear(posts: Post[]): Array<{ year: number; posts: Post[] }> {
  const byYear = new Map<number, Post[]>();

  for (const post of posts) {
    const year = post.data.pubDate.getFullYear();
    const bucket = byYear.get(year);
    if (bucket) bucket.push(post);
    else byYear.set(year, [post]);
  }

  return [...byYear.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([year, entries]) => ({ year, posts: entries }));
}

/** Previous/next neighbours for in-post navigation (list is newest-first). */
export function getNeighbours(posts: Post[], id: string): { newer?: Post; older?: Post } {
  const index = posts.findIndex((post) => post.id === id);
  if (index === -1) return {};
  return { newer: posts[index - 1], older: posts[index + 1] };
}
