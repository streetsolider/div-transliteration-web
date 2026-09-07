// Node harness: run the exported tiny model through the app's chunking with
// Transformers.js, score it on a TSV of (latin, keymap) rows, and time it.
//
//   node harness.mjs <export_dir> <rows.tsv> [--dtype q8|fp32|fp16] [--n 500]
//
// rows.tsv: header "latin\tkeymap" then one pair per line, targets in the Segha keymap (ASCII).
// Prints exact / char accuracy against keymap targets and ms per chunk, so the
// same numbers as decode_eval.py can be compared with the browser build.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { env, Tensor, AutoModelForSeq2SeqLM } from '@huggingface/transformers';
import { transliterate, makeKeymapToThaana, byt5Encode, byt5Decode, maxNewTokensFor } from './pipeline.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const exportDir = path.resolve(args[0]);
const tsv = path.resolve(args[1]);
const dtype = args.includes('--dtype') ? args[args.indexOf('--dtype') + 1] : 'q8';
const n = args.includes('--n') ? parseInt(args[args.indexOf('--n') + 1], 10) : 500;

env.allowRemoteModels = false;
env.localModelPath = path.dirname(exportDir);
const modelId = path.basename(exportDir);

const model = await AutoModelForSeq2SeqLM.from_pretrained(modelId, { dtype });
const keymapToThaana = makeKeymapToThaana(JSON.parse(readFileSync(path.join(here, 'keymap.json'), 'utf8')));

let chunkCount = 0, genMs = 0;
async function generate(chunks) {
  const t0 = performance.now();
  const enc = byt5Encode(chunks, Tensor);
  const out = await model.generate({ ...enc, max_new_tokens: maxNewTokensFor(chunks), num_beams: 1, do_sample: false });
  const dec = byt5Decode(out);
  genMs += performance.now() - t0;
  chunkCount += chunks.length;
  return dec;
}

function charAcc(p, l) {
  if (!p && !l) return 1;
  if (!p || !l) return 0;
  const m = l.length, nn = p.length;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= nn; i++) {
    const cur = [i];
    for (let j = 1; j <= m; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p[i - 1] === l[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return 1 - prev[m] / Math.max(nn, m);
}

const rows = readFileSync(tsv, 'utf8').split('\n').slice(1).filter(Boolean).slice(0, n)
  .map((l) => { const [latin, keymap] = l.split('\t'); return { latin, keymap }; });
// warm-up
await generate(['test']);
chunkCount = 0; genMs = 0;

let exact = 0, chars = 0;
const t0 = performance.now();
for (const r of rows) {
  // score on raw keymap output, chunk-free, the way decode_eval does
  const [km] = await generate([r.latin]);
  if (km === r.keymap) exact++;
  chars += charAcc(km, r.keymap);
}
const wall = performance.now() - t0;
console.log(JSON.stringify({
  model: modelId, dtype, rows: rows.length,
  exact: +(exact / rows.length).toFixed(4), char_acc: +(chars / rows.length).toFixed(4),
  ms_per_row: +(genMs / chunkCount).toFixed(1), wall_s: +(wall / 1000).toFixed(1),
}));
// one end-to-end sample through the chunk pipeline
const sample = rows[0].latin;
console.log('pipeline sample:', sample, '=>', await transliterate(sample, generate, keymapToThaana));
