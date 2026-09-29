/**
 * Turns a standalone image paragraph into a `<figure>` with a `<figcaption>`.
 *
 *   ![Alt text](./photo.jpg "Caption shown under the image")
 *
 * A paragraph whose only child is an image becomes the figure; the image's
 * Markdown title is used as the caption, falling back to nothing (the alt text
 * stays on the `<img>` for screen readers either way).
 *
 * This runs on the mdast side and only rewrites the wrapper, so the image node
 * itself is untouched and Astro's build-time image optimization still applies.
 */
export const figurePlugin = {
  name: 'figure',

  paragraph(node, ctx) {
    const children = node.children ?? [];
    if (children.length !== 1) return;

    const image = children[0];
    if (!image || image.type !== 'image') return;

    ctx.setProperty(node, 'data', {
      hName: 'figure',
      hProperties: { class: 'figure' },
    });

    const caption = typeof image.title === 'string' ? image.title.trim() : '';
    if (!caption) return;

    ctx.appendChild(node, {
      type: 'paragraph',
      data: { hName: 'figcaption', hProperties: { class: 'figure-caption' } },
      children: [{ type: 'text', value: caption }],
    });
  },
};
