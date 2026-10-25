# Instructions
For each item, create a side branch and implement it. Do not merge anything.
Name each branch in the form `branch-title` (do not include the TODO number in the branch name).

Any new features added should have an accompanying API access point (if applicable).

Skip anything mared as "[DONE]".

# TODO
1. Add whisper model to the app.
   - make this another tab, after the "Watermark" tab
   - just have it be a simple whisper interface, where one can upload/record audio, and have it transcripe to text the user can copy/paste
2. Add various presets for the generation settings to mimic emotion.
3. Add ability to convert the dialog to another language (if this doesn't already exist).
   - for example, user wants to do a voice conversion AND an on-the-fly translation like english -> new voice -> japanese dialect
4. Add ability to use an AI model to remove background noise from a reference clip.
   - for example, the reference clip is in a noisy room, and they want to isolate the person talking to get a cleaner reference clip for audio generation