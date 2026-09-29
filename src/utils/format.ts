/**
 * Date and reading-time helpers shared by the listing and post pages.
 */

/*
 * Dates are always formatted in English, even for Chinese posts, so the site has
 * one consistent date style rather than switching between `2025年2月24日` and
 * `February 24, 2025`. A post's `lang` field still drives its `lang` attribute,
 * CJK typography, and how its headings are slugged.
 *
 * The formatter is built once and reused: `Intl.DateTimeFormat` construction is
 * comparatively expensive and this runs for every post on every listing.
 */
const DATE_FORMAT = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

/** Human-readable publication date, always English. */
export function formatDate(date: Date): string {
  return DATE_FORMAT.format(date);
}

/** `2024-03-08`, for `<time datetime>` and sorting-sensitive display. */
export function isoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/**
 * Rough reading time from the raw Markdown body.
 *
 * CJK is counted per character (~400 chars/min) and Latin script per word
 * (~220 words/min), so a mixed-language post is measured correctly without being
 * told which language it is in. Code fences and math are stripped first so a
 * long listing doesn't inflate the estimate.
 */
export function readingTime(body: string | undefined): number {
  if (!body) return 1;

  const text = body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!?\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/^---[\s\S]*?^---/m, ' ');

  const cjk = (text.match(/[\u4e00-\u9fff\u3040-\u30ff]/g) ?? []).length;
  const words = (text.replace(/[\u4e00-\u9fff\u3040-\u30ff]/g, ' ').match(/[A-Za-z0-9'’-]+/g) ?? [])
    .length;

  const minutes = cjk / 400 + words / 220;
  return Math.max(1, Math.round(minutes));
}

/**
 * `5 min read`. Always English, for the same reason as `formatDate` — the post
 * metadata line stays consistent instead of mixing scripts within one line.
 */
export function readingTimeLabel(minutes: number): string {
  return `${minutes} min read`;
}
