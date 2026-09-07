// Chunk-and-stitch pipeline, a faithful port of app.py's transliterate():
// paragraphs -> sentences ([.!?]) -> phrases ([,;]) -> 10-word chunks, one
// batched generate per paragraph, stitched back with the original punctuation,
// keymap -> Thaana, ASCII -> Arabic punctuation. Works in Node and the browser.

export const CHUNK_WORDS = 10;

export function splitIntoWordChunks(text, maxWords = CHUNK_WORDS) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(' '));
    if (i + maxWords >= words.length) break;
  }
  return chunks;
}

// ByT5 tokenizer, which needs no vocabulary: id = utf-8 byte + 3, with
// pad=0, eos=1, unk=2. Transformers.js has no tokenizer.json for it, so it is
// done here in a few lines; the output side is ASCII keymap, so decoding is
// bytes back to text.
const PAD = 0, EOS = 1, OFFSET = 3;
const enc = new TextEncoder(), dec = new TextDecoder();

export function byt5Encode(texts, Tensor, maxLength = 256) {
  const rows = texts.map((t) => {
    const b = Array.from(enc.encode(t)).slice(0, maxLength - 1).map((x) => x + OFFSET);
    b.push(EOS);
    return b;
  });
  const L = Math.max(...rows.map((r) => r.length));
  const ids = new BigInt64Array(rows.length * L);
  const mask = new BigInt64Array(rows.length * L);
  rows.forEach((r, i) => r.forEach((v, j) => { ids[i * L + j] = BigInt(v); mask[i * L + j] = 1n; }));
  return {
    input_ids: new Tensor('int64', ids, [rows.length, L]),
    attention_mask: new Tensor('int64', mask, [rows.length, L]),
  };
}

// Runaway guard (QC finding): a rare name can send the decoder into a loop to
// the hard limit. Keymap is ~1.05-1.2x the Latin byte length, so 2.5x + 16 is
// generous for any real output and still stops a loop early.
export function maxNewTokensFor(chunks, hardCap = 256) {
  const longest = Math.max(...chunks.map((c) => enc.encode(c).length));
  return Math.min(hardCap, Math.ceil(2.5 * longest) + 16);
}

export function byt5Decode(outputTensor) {
  const [n, L] = outputTensor.dims;
  const data = outputTensor.data;
  const out = [];
  for (let i = 0; i < n; i++) {
    const bytes = [];
    for (let j = 0; j < L; j++) {
      const v = Number(data[i * L + j]);
      if (v === EOS) break;
      if (v >= OFFSET && v < OFFSET + 256) bytes.push(v - OFFSET);
    }
    out.push(dec.decode(Uint8Array.from(bytes)));
  }
  return out;
}

export function makeKeymapToThaana(table) {
  return (s) => Array.from(s, (c) => table[c] ?? c).join('');
}

// plan a paragraph: returns {chunks, plan}
export function planParagraph(paragraph) {
  let sentences = paragraph.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
  sentences = sentences.map((s) => s.trim()).filter(Boolean);
  if (!sentences.length) sentences = [paragraph];
  const chunks = [];
  const plan = [];
  for (const sentence of sentences) {
    let endingPunct = '';
    let sentenceText = sentence;
    if (sentence && '.!?'.includes(sentence[sentence.length - 1])) {
      endingPunct = sentence[sentence.length - 1];
      sentenceText = sentence.slice(0, -1).trim();
    }
    let phrases = sentenceText.match(/[^,;]+[,;]?/g) || [];
    phrases = phrases.map((p) => p.trim()).filter(Boolean);
    const phrasePlans = [];
    for (const phrase of phrases) {
      let delimiter = '';
      let phraseText = phrase;
      if (phrase && ',;'.includes(phrase[phrase.length - 1])) {
        delimiter = phrase[phrase.length - 1];
        phraseText = phrase.slice(0, -1).trim();
      }
      const parts = phraseText.split(/\s+/).filter(Boolean).length > CHUNK_WORDS
        ? splitIntoWordChunks(phraseText) : [phraseText];
      const start = chunks.length;
      chunks.push(...parts);
      phrasePlans.push({ delimiter, start, end: chunks.length });
    }
    plan.push({ endingPunct, phrases: phrasePlans });
  }
  return { chunks, plan };
}

export function stitchParagraph(decoded, plan) {
  const out = [];
  for (const sent of plan) {
    const parts = [];
    for (const ph of sent.phrases) {
      let phraseThaana = decoded.slice(ph.start, ph.end).join(' ');
      if (ph.delimiter) phraseThaana += ph.delimiter;
      parts.push(phraseThaana);
    }
    let s = parts.join(' ');
    if (sent.endingPunct) s += sent.endingPunct;
    s = s.replace(/,/g, '،').replace(/;/g, '؛').replace(/\?/g, '؟');
    out.push(s);
  }
  return out.join(' ');
}

// generate(chunks) -> array of keymap strings, provided by the caller
export async function transliterate(text, generate, keymapToThaana) {
  const paragraphs = text.split('\n\n');
  const outParas = [];
  for (const para of paragraphs) {
    if (!para.trim()) { outParas.push(''); continue; }
    const { chunks, plan } = planParagraph(para);
    if (!chunks.length) { outParas.push(''); continue; }
    const keymaps = await generate(chunks);
    outParas.push(stitchParagraph(keymaps.map(keymapToThaana), plan));
  }
  return outParas.join('\n\n');
}
