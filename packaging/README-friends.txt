Poly Voice Changer
==================

Talk into your microphone and other apps (Discord, games, OBS) hear you in another voice.

You need
--------
* Windows 10 or 11 with an NVIDIA graphics card (GTX 16xx / RTX or newer) and an up-to-date driver.
* VB-CABLE, a free virtual audio cable: https://vb-audio.com/Cable/
  Download it, unzip, right-click VBCABLE_Setup_x64.exe > Run as administrator > Install, then restart.

Setup (once)
------------
1. Unzip this whole folder somewhere permanent (e.g. Documents\PolyVoiceChanger). Keep all files together.
2. Run PolyVoiceChanger.exe.
3. Click "Add voice...", type a name, choose a clip of the voice (5-60 seconds of one person talking
   clearly, no music) and Save.

Every time
----------
1. Pick your voice, your Microphone, and "CABLE Input" as the Output.
2. Press Start. The first start downloads the voice model (about 1 GB), so it takes a few minutes.
   Then stay quiet for a second while it measures your background noise.
3. In Discord: Settings > Voice & Video > Input Device = "CABLE Output (VB-Audio Virtual Cable)".
   Turn off Discord's noise suppression ("Krisp") if the voice sounds choppy.
4. Talk, then pause. Each phrase is heard about a second after you stop talking.

Tips
----
* It picks up everything above the "speech threshold": if it triggers on background noise, enter a
  higher threshold under Advanced (e.g. 0.03); if it misses quiet speech, a lower one (e.g. 0.01).
* To hear yourself: Windows Sound settings > Recording > CABLE Output > Properties > Listen >
  "Listen to this device".
* Voices, settings and a log file are kept in %LOCALAPPDATA%\PolyVoiceChanger.
  The model is cached in %USERPROFILE%\.cache\huggingface.
* Only use voices you have permission to use.
