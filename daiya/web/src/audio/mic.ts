export interface MicHandle {
  /** Actual AudioContext sample rate (the worklet resamples to 16 kHz regardless). */
  sampleRate: number;
  stop(): void;
}

/**
 * Capture the browser mic and deliver 16 kHz mono PCM int16 frames (100 ms
 * ArrayBuffers) ready to be sent as binary WebSocket messages.
 */
export async function startMic(onFrame: (frame: ArrayBuffer) => void): Promise<MicHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Microphone access needs a secure context — open this page over HTTPS or localhost.');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ sampleRate: 16000 });
  } catch {
    ctx = new AudioContext(); // worklet resamples when the context rate differs
  }

  try {
    await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}worklets/pcm-worklet.js`);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close();
    throw err;
  }

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'pcm-worklet', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => onFrame(e.data);
  source.connect(node);
  node.connect(ctx.destination); // keeps the node pulled; the worklet outputs silence
  if (ctx.state === 'suspended') await ctx.resume();

  return {
    sampleRate: ctx.sampleRate,
    stop() {
      node.port.onmessage = null;
      source.disconnect();
      node.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
