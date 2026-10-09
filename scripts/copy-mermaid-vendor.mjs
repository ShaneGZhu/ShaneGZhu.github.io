/**
 * Copies the browser mermaid bundle into public/vendor/ so pages can load it
 * without going through the Vite client bundle (mermaid's ESM graph is heavy
 * and does not always resolve under Rolldown). The file is generated, not
 * committed — run this via the `prebuild` script.
 */
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js');
const dest = join(root, 'public', 'vendor', 'mermaid.min.js');

mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
console.log('copied mermaid.min.js -> public/vendor/');
