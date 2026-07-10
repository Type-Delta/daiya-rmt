# AGENTS.md

Guide for agentic coding agents working in the `Daiya-RMT` repository.

## About Daiya-RMT

Daiya-RMT, short for 'Daiya' is a research project focused on developing a speaker and context aware transcription system that also supports mixed-lingual conversations. The main focus of the project is to be able to transcribe realworld technical/casual conversations in Thai-English and Japanese-English, which are common in both countries.

### Goal

- An API server and CLI (share the same backend) that can transcribe conversations in Thai-English and Japanese-English, with speaker diarization and context awareness.
- The API should support streaming transcription with follow-up corrections ith speaker diarization and context awareness.
- The CLI should be able to display the transcription in real-time, with speaker diarization and context awareness.
- Transcription:
   - Transcription should be accurate (with follow-up corrections) and low-latency, suitable for real-time applications.
   - Can add custom context and terminology to the system to improve accuracy and relevance of the transcription.
- Speaker diarization:
   - The system should be able to identify and differentiate between multiple speakers in a conversation, and is able to retain speaker identification across multiple processing turns. This is to make sure that the system can handle smaller segments of conversations without losing track of who is speaking.

### Architecture

Daiya can be broken down into four main components:
1. **Transcription Engine**: This component is responsible for converting audio input into text. It should support both Thai-English and Japanese-English mixed-lingual conversations, and be able to handle technical and casual language.
2. **Speaker Diarization Engine**: This component is responsible for identifying and differentiating between multiple speakers in a conversation. It should be able to retain speaker identification across multiple processing turns.
3. **Multiplexer**: This component is responsible for managing the flow of data between the Transcription Engine and the Speaker Diarization Engine, ensuring that output fragments from both components align correctly. It also handles follow-up corrections and context management to improve transcription accuracy over time.
4. **Interface Layer**: This component provides the API server and CLI for users to interact with the system. It should support streaming transcription and display results in real-time.

### How it could works (theoretically)

Tinfoil Hats on, because this is just a theoretical design and implementation plan for the system, it will change as we working on the project. (remember, this is a research project, not a production system, yet!)

To achieve the goals outlined above, we need to solve the following challenges:
1. **Transcription Speed**: we need a fast transcription engine for low-latency transcription.
2. **Mixed-lingual Support**: we need a transcription engine that can handle both Thai-English and Japanese-English mixed-lingual conversations, which can be challenging due to the differences in language structure, vocabulary and ambiguous word boundaries.
3. **Stateful Speaker Diarization**: we need a speaker diarization engine that can retain speaker identification across multiple processing turns to be able to handle smaller segments of conversations without losing track of who is speaking.
4. **Context Awareness**: we need a system that can manage context and terminology to improve transcription accuracy and relevance, especially for technical conversations.
5. **Real-time Streaming**: we need to design the system to support real-time (or near real-time) streaming transcription with follow-up corrections.

To address these challenges, we can consider the following approaches for each challenge:
1. **Transcription Speed**: we can use a lightweight and efficient transcription model, output from this model can be used as a preliminary transcription result, which can be further refined with follow-up corrections to improve accuracy.
   Currently, we are looking into using 2 or 3 pass transcription.
   - **3-pass transcription**: the first pass is a small and fast transcription model that produces a preliminary transcription result, the second pass is a Large transcription model that replaces the preliminary transcription result, and the third pass will combine the transcription from second pass with conversation context/terminology to produce a more accurate transcription.
   - **2-pass transcription**: similar to 3-pass transcription, but we skip the first pass and just use the Large transcription model for the second pass, and then combine it with conversation context/terminology for the final transcription result.
   If the Large transcription model is fast enough, we can skip the first pass and just use 2-pass transcription, but if not, we can use 3-pass transcription to achieve a good balance between speed and accuracy.
2. **Mixed-lingual Support**: we can fine-tune a multilingual transcription model (Whisper) using LoRA on a custom dataset of Thai-English and Japanese-English mixed-lingual conversations, which can help the model learn to handle the unique challenges of mixed-lingual transcription.
3. **Stateful Speaker Diarization**: we can modify an existing speaker diarization model (pyannote) to retain speaker identification across multiple processing turns, which can be achieved by caching vector representing speaker characteristics and using them to match speakers in subsequent turns.
4. **Context Awareness**: we can use small LLMs to verify and correct the transcription results based on the context and terminology provided by the user, plus summary of the current conversation direction, which can help improve the accuracy and relevance of the transcription, especially for technical conversations, or multi subject conversations where the subject can change frequently. Ideally, the LLM doesn't need to understand the full conversation, just the current direction of the conversation should be sufficient.
5. **Real-time Streaming**: we can design split input audio into smaller segments (we could try Silero VAD for this) so we can stream output in segments instead of waiting for the whole conversation to be processed, and use a multiplexer to manage the flow of data between the Transcription Engine and the Speaker Diarization Engine, ensuring that output fragments from both components align correctly, and also handle follow-up corrections and context management to improve transcription accuracy over time.

This research aims to answer the following questions:
1. [x] Can we modify pyannote to retain speaker identification across multiple processing turns without compromising the accuracy? *Yes! We can and we don't need to modify pyannote*
2. [x] Can we make speaker diarization works in near-real-time? *Yes! We can but needs some more realworld testing*
3. [x] Can the fine-tuned Whisper model achieve the desired accuracy for Thai-English and Japanese-English mixed-lingual transcription? and if so which base model is the best for this? *Yes! The fine-tuned Whisper model can achieve the desired accuracy*
4. [ ] What is more efficient between using 2 vs 3 pass transcription, or is there a better way?
5. [ ] What is the best LLM model for verifying and correcting transcription results, that is fast and accurate enough for this task?
6. [ ] What is the best segment size for streaming transcription, that doesn't compromise the accuracy of Speaker Diarization while still providing low-latency transcription?

## Working on this project

- When training the models, spawn a training process completely isolated from the harness, so the training process can continue running even if the harness is restarted or killed. To monitor the training process, use a separate process tied to the harness that can monitor the training process and report its status back once the traning has reached a certain milestone/killed or completed. Never monitor the training directly yourself.
