/**
 * Date and reading-time helpers shared by the listing and post pages.
 */

const FORMATTERS = {
  en: new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
  zh: new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }),
};

/** Human-readable date in the post's own language. */
export function formatDate(date: Date, lang: 'en' | 'zh' = 'en'): string {
  return (FORMATTERS[lang] ?? FORMATTERS.en).format(date);
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

/** `5 min read` / `5 分钟阅读`, matching the post's language. */
export function readingTimeLabel(minutes: number, lang: 'en' | 'zh' = 'en'): string {
  return lang === 'zh' ? `${minutes} 分钟阅读` : `${minutes} min read`;
}
