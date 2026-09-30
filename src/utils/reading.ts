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

/** Every tag used by the reading list, most-used first then alphabetical. */
export function collectReadingTags(entries: ReadingEntry[]): string[] {
  const counts = new Map<string, number>();

  for (const entry of entries) {
    for (const tag of entry.data.tags) {
      const label = tag.trim();
      if (label) counts.set(label, (counts.get(label) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([label]) => label);
}
