/**
 * Single place to edit site-wide metadata.
 *
 * `url` must match the deployed origin for canonical links, the sitemap and the
 * RSS feed to be correct. For a GitHub user/organization Pages repository
 * (`<user>.github.io`) the site is served from the domain root, so `base` stays
 * undefined.
 */
export const SITE = {
  url: 'https://shanegzhu.github.io',
  title: 'Shane G. Zhu',
  tagline: 'Notes on systems, inference, and the machinery underneath.',
  description:
    'Long-form notes on distributed systems, LLM inference engines, and performance engineering — by Shane G. Zhu.',
  author: 'Shane G. Zhu',
  // Used as the default `lang` attribute; individual posts can override it.
  defaultLanguage: 'en',
  locale: 'en_US',
  postsPerPage: 8,
};

export const NAV = [
  { label: 'Writing', href: '/posts' },
  { label: 'Archive', href: '/archive' },
  { label: 'Tags', href: '/tags' },
  { label: 'About', href: '/about' },
];

export const SOCIAL = [
  { label: 'GitHub', href: 'https://github.com/ShaneGZhu' },
  { label: 'RSS', href: '/rss.xml' },
];
