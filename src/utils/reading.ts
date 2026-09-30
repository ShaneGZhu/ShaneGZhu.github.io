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
