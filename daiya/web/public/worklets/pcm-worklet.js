// Resamples the mic input to 16 kHz mono and posts Int16Array frames
// (100 ms / 1600 samples each) to the main thread for the /ws/stream socket.
// Linear interpolation; identity pass-through when the context already runs
// at 16 kHz (Chrome/Firefox honor `new AudioContext({sampleRate: 16000})`).

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 1600; // 100 ms

class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.pos = 0; // fractional read cursor into the current block; -1 addresses `tail`
    this.tail = 0; // last sample of the previous block, for interpolation across blocks
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.n = 0;
  }

  process(inputs) {
    const src = inputs[0][0];
    if (!src || src.length === 0) return true;

    let pos = this.pos;
    while (pos <= src.length - 1) {
      const i0 = Math.floor(pos);
      const t = pos - i0;
      const a = i0 < 0 ? this.tail : src[i0];
      const b = i0 + 1 < src.length ? src[i0 + 1] : a;
      const s = Math.max(-1, Math.min(1, a + (b - a) * t));
      this.frame[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === FRAME_SAMPLES) {
        const out = this.frame.slice();
        this.port.postMessage(out.buffer, [out.buffer]);
        this.n = 0;
      }
      pos += this.ratio;
    }
    this.pos = pos - src.length;
    this.tail = src[src.length - 1];
    return true;
  }
}

registerProcessor('pcm-worklet', PcmWorklet);
