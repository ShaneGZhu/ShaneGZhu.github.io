/**
 * Single place to edit site-wide metadata.
 *
 * `url` must match the deployed origin for canonical links and the sitemap to be
 * correct. For a GitHub user/organization Pages repository (`<user>.github.io`)
 * the site is served from the domain root, so `base` stays undefined.
 */
export const SITE = {
  url: 'https://shanegzhu.github.io',
  title: 'Shengguang Zhu',
  description:
    'ML systems engineer working on RL infrastructure and high-throughput LLM inference, contributing to SGLang and FastDeploy. Notes on schedulers, kernels, and the systems side of machine learning.',
  author: 'Shengguang Zhu',
  /** Shown as a subtitle under the name. */
  role: 'ML Systems Engineer · High-Performance Computing',
  email: 'shengguangzhu@qq.com',
  // Used as the default `lang` attribute; individual posts can override it.
  defaultLanguage: 'en',
  locale: 'en_US',
  postsPerPage: 8,
  /** How many posts the home page lists before linking to the full index. */
  postsOnHome: 4,
};

export const NAV = [
  { label: 'Home', href: '/blog/' },
  { label: 'Writing', href: '/blog/posts/' },
  { label: 'Reading', href: '/blog/reading/' },
  { label: 'Archive', href: '/blog/archive/' },
  { label: 'Tags', href: '/blog/tags/' },
  { label: 'About', href: '/blog/about/' },
];

/**
 * `icon` names a glyph in `src/components/Icon.astro`. Icons appear in the footer,
 * on the home page, and on the About page, so this set has to cover whatever is
 * listed here.
 *
 * `@type` keeps each icon name a literal so a typo is a type error rather than a
 * silently missing glyph. (`as const` is not available — this is a `.mjs` file.)
 *
 * @type {ReadonlyArray<{ label: string, href: string, icon: 'github' | 'zhihu' | 'xiaohongshu' | 'mail' }>}
 */
export const SOCIAL = [
  { label: 'GitHub', href: 'https://github.com/ShaneGZhu', icon: 'github' },
  {
    label: 'Zhihu',
    href: 'https://www.zhihu.com/people/zhu-sheng-guang-30',
    icon: 'zhihu',
  },
  {
    label: 'Rednote',
    href: 'https://www.xiaohongshu.com/user/profile/5c596598000000001200e203',
    icon: 'xiaohongshu',
  },
  { label: 'Email', href: 'mailto:shengguangzhu@qq.com', icon: 'mail' },
];

/** What I work on — shown on the home page and the About page. */
export const FOCUS = [
  {
    title: 'RL Infrastructure',
    body: 'Building and optimizing reinforcement learning training and inference pipelines.',
  },
  {
    title: 'LLM Inference',
    body: 'Contributing to SGLang and FastDeploy, across the serving runtime and the kernels underneath it.',
    links: [
      { label: 'SGLang', href: 'https://github.com/sgl-project/sglang' },
      { label: 'FastDeploy', href: 'https://github.com/PaddlePaddle/FastDeploy' },
    ],
  },
  {
    title: 'Kernel Optimization',
    body: 'Designing and tuning high-performance custom kernels.',
  },
  {
    title: 'Writing',
    body: 'Maintaining MLsys-Note, an open space for notes and code on AI infrastructure.',
    links: [{ label: 'MLsys-Note', href: 'https://github.com/ShaneGZhu/MLsys-Note' }],
  },
];
