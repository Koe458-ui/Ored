#!/usr/bin/env node

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import sharp from 'sharp';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = (name) => join(root, name);

// The mark: two arcs of a ring, orange lower-left, red upper-right, on a 512 grid
const MARK =
  '<path fill="#FF6A00" d="M405.30 433.58A232 232 0 0 1 78.42 106.70L163.93 192.22A112 112 0 0 0 319.78 348.07Z"/>' +
  '<path fill="#FF1F2D" d="M106.70 78.42A232 232 0 0 1 433.58 405.30L348.07 319.78A112 112 0 0 0 192.22 163.93Z"/>';

const SIZE = 512;
const RING = 232;

const ICON = 0.84;
const SAFE = 0.8;

if (SAFE * RING > 0.4 * SIZE) {
  throw new Error(`SAFE ${SAFE} puts the ring ${(SAFE * RING).toFixed(1)} units from centre; ` +
    `the maskable safe circle is ${0.4 * SIZE}.`);
}

function svg({ scale = 1, ground = false } = {}) {
  const half = SIZE / 2;
  const bg = ground ? `<rect width="${SIZE}" height="${SIZE}" fill="#FFFFFF"/>` : '';
  const body = scale === 1
    ? MARK
    : `<g transform="translate(${half} ${half}) scale(${scale}) translate(${-half} ${-half})">${MARK}</g>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SIZE} ${SIZE}">${bg}${body}</svg>`;
}

const tabSvg = svg();
const iconSvg = svg({ scale: ICON, ground: true });
const maskSvg = svg({ scale: SAFE, ground: true });

writeFileSync(out('favicon.svg'), tabSvg);
writeFileSync(out('logo.svg'), tabSvg + '\n');

const render = (source, px) =>
  sharp(Buffer.from(source))
    .resize(px, px, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png({ compressionLevel: 9, palette: false });

const flat = (source, px) => render(source, px).flatten({ background: '#ffffff' });

await flat(iconSvg, 192).toFile(out('icon-192.png'));
await flat(iconSvg, 512).toFile(out('icon-512.png'));
await flat(iconSvg, 180).toFile(out('apple-touch-icon.png'));
await flat(maskSvg, 512).toFile(out('icon-maskable-512.png'));

const ICO_SIZES = [16, 32, 48, 64, 128];
const frames = await Promise.all(
  ICO_SIZES.map((px) => render(tabSvg, px).toBuffer())
);

const header = Buffer.alloc(6);
header.writeUInt16LE(0, 0);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(frames.length, 4);

const dir = Buffer.alloc(16 * frames.length);
let offset = header.length + dir.length;
frames.forEach((png, i) => {
  const at = i * 16;
  dir.writeUInt8(ICO_SIZES[i] >= 256 ? 0 : ICO_SIZES[i], at);
  dir.writeUInt8(ICO_SIZES[i] >= 256 ? 0 : ICO_SIZES[i], at + 1);
  dir.writeUInt8(0, at + 2);
  dir.writeUInt8(0, at + 3);
  dir.writeUInt16LE(1, at + 4);
  dir.writeUInt16LE(32, at + 6);
  dir.writeUInt32LE(png.length, at + 8);
  dir.writeUInt32LE(offset, at + 12);
  offset += png.length;
});

writeFileSync(out('favicon.ico'), Buffer.concat([header, dir, ...frames]));

console.log('logo.svg, favicon.svg, favicon.ico (%s), icon-192, icon-512, icon-maskable-512, apple-touch-icon',
  ICO_SIZES.join('/'));
