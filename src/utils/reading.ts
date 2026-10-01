/**
 * Reading-list helpers.
 *
 * The `reading` collection is other people's writing, kept apart from `posts` so
 * the article listings stay a record of what I wrote.
 */
import { getCollection, type CollectionEntry } from 'astro:content';

export type ReadingEntry = CollectionEntry<'reading'>;

/** Collected items, most recently added first. */
export async function getReadingList(): Promise<ReadingEntry[]> {
  const entries = await getCollection('reading');
  return entries.sort((a, b) => b.data.added.valueOf() - a.data.added.valueOf());
}

/**
 * The source's hostname, for display next to the title.
 *
 * Returns the raw value if it will not parse, so a malformed URL shows up in the
 * list rather than silently rendering nothing.
 */
export function sourceHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

/**
 * Reading entries carrying a given tag, most recently added first.
 *
 * The tag arrives as a URL segment, so it is a plain string while `tags` is typed
 * from an enum; the widening cast is what lets the two meet.
 */
export function readingWithTag(entries: ReadingEntry[], tag: string): ReadingEntry[] {
  return entries.filter((entry) => (entry.data.tags as readonly string[]).includes(tag));
}

/**
 * The distinct tags used across the reading list.
 *
 * Returned as the tag labels themselves, which double as the values in the
 * `tags` enum, so a tag page can be built for a tag that only appears here.
 */
export function collectReadingTags(entries: ReadingEntry[]): string[] {
  const seen = new Set<string>();
  for (const entry of entries) for (const tag of entry.data.tags) seen.add(tag);
  return [...seen].sort();
}
