/**
 * Regenerates the favicon set in `public/` from the avatar.
 *
 *   node scripts/make-favicons.mjs
 *
 * Public pages reference these files by path, so a normal build never needs this
 * script. It exists so the icons can be reproduced after the avatar changes
 * instead of being opaque binaries nobody dares to touch.
 *
 * The avatar is 224px square, so it is upscaled for the 256px output; that is
 * acceptable for a home-screen icon and keeps the source at the size the page
 * actually displays.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import sharp from 'sharp';

const ROOT = path.resolve(import.meta.dirname, '..');
const SOURCE = path.join(ROOT, 'src/assets/avatar.jpg');
const OUT_DIR = path.join(ROOT, 'public');

/** Sizes bundled into the multi-resolution `favicon.ico`. */
const ICO_SIZES = [16, 32, 48];

/**
 * Builds a multi-image `.ico` from PNG buffers.
 *
 * The container is a 6-byte header, then one 16-byte directory entry per image,
 * then the payloads. Browsers accept PNG data inside an ICO, which keeps this to
 * a few lines instead of hand-rolling a BMP encoder.
 *
 * @param {{ size: number, data: Buffer }[]} images
 */
function buildIco(images) {
  const HEADER = 6;
  const ENTRY = 16;

  const header = Buffer.alloc(HEADER);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: 1 = icon
  header.writeUInt16LE(images.length, 4);

  let offset = HEADER + ENTRY * images.length;

  const entries = images.map(({ size, data }) => {
    const entry = Buffer.alloc(ENTRY);
    // 256px and above are encoded as 0 in this byte.
    entry.writeUInt8(size >= 256 ? 0 : size, 0); // width
    entry.writeUInt8(size >= 256 ? 0 : size, 1); // height
    entry.writeUInt8(0, 2); // palette size
    entry.writeUInt8(0, 3); // reserved
    entry.writeUInt16LE(1, 4); // colour planes
    entry.writeUInt16LE(32, 6); // bits per pixel
    entry.writeUInt32LE(data.length, 8);
    entry.writeUInt32LE(offset, 12);
    offset += data.length;
    return entry;
  });

  return Buffer.concat([header, ...entries, ...images.map((image) => image.data)]);
}

/**
 * A photo compresses far better as an indexed PNG, but below ~48px the palette
 * overhead makes it larger than truecolour, so the tiny sizes stay truecolour.
 */
async function png(size, { flatten = false } = {}) {
  let pipeline = sharp(SOURCE).resize(size, size, { fit: 'cover', position: 'center' });
  // iOS does not composite alpha onto a home screen icon.
  if (flatten) pipeline = pipeline.flatten({ background: '#ffffff' });

  return pipeline
    .png(
      size >= 48
        ? { compressionLevel: 9, palette: true, effort: 10 }
        : { compressionLevel: 9 },
    )
    .toBuffer();
}

const files = [
  [180, 'apple-touch-icon.png', { flatten: true }],
  [256, 'icon-256.png', {}],
];

await mkdir(OUT_DIR, { recursive: true });

const icoImages = await Promise.all(
  ICO_SIZES.map(async (size) => ({ size, data: await png(size) })),
);
const ico = buildIco(icoImages);
await writeFile(path.join(OUT_DIR, 'favicon.ico'), ico);
console.log(`favicon.ico          ${ICO_SIZES.join('/')}px  ${(ico.length / 1024).toFixed(1)} KiB`);

for (const [size, name, options] of files) {
  const data = await png(size, options);
  await writeFile(path.join(OUT_DIR, name), data);
  console.log(`${name.padEnd(20)} ${size}px      ${(data.length / 1024).toFixed(1)} KiB`);
}
