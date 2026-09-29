import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';

import { getPublishedPosts } from '../utils/posts';
import { SITE } from '../site.config.mjs';

/**
 * RSS 2.0 feed at `/rss.xml`.
 *
 * Only metadata is published, not full post bodies: the compiled HTML contains
 * optimized image URLs and KaTeX markup that depend on the site's stylesheet, so
 * it reads poorly in a reader. Each item links back to the canonical post.
 */
export const GET: APIRoute = async (context) => {
  const posts = await getPublishedPosts();

  return rss({
    title: SITE.title,
    description: SITE.description,
    site: context.site ?? SITE.url,
    trailingSlash: false,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: `/posts/${post.id}`,
      categories: post.data.tags,
      author: SITE.author,
    })),
    customData: `<language>${SITE.defaultLanguage}</language>`,
  });
};
