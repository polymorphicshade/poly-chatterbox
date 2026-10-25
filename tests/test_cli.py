import json

import pytest
import soundfile as sf

from poly_chatterbox.cli import main


def run(capsys, *argv):
    code = main(["--backend", "mock", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_tts_commands(engine, tmp_path, ref_wav, capsys):
    for args in (
        ["original", "Hello", "--exaggeration", "0.7", "--cfg-weight", "0.3"],
        ["turbo", "Hi [laugh]", "--ref-audio", str(ref_wav), "--top-k", "200", "--no-norm-loudness"],
        ["nano", "Hi", "--voice", "podcast"],
        ["multilingual", "Bonjour", "-l", "fr", "--model-version", "v2"],
    ):
        out = tmp_path / f"{args[0]}.wav"
        code, stdout, err = run(capsys, *args, "-o", str(out), "--json")
        assert code == 0, err
        assert json.loads(stdout)["mode"] == args[0]
        assert sf.info(str(out)).samplerate == 24000


def test_model_specific_options(capsys):
    with pytest.raises(SystemExit):
        main(["turbo", "--help"])
    out = capsys.readouterr().out
    assert "--top-k" in out and "--exaggeration" not in out and "(default: 0.95)" in out
    with pytest.raises(SystemExit):
        main(["original", "Hi", "--top-k", "5"])


def test_convert_and_library(engine, tmp_path, ref_wav, capsys):
    out = tmp_path / "vc.wav"
    code, _, err = run(capsys, "convert", str(ref_wav), "--target-audio", str(ref_wav), "-o", str(out))
    assert code == 0 and out.is_file(), err
    code, stdout, _ = run(capsys, "voices", "add", "CLI Voice", "--ref-audio", str(ref_wav), "-l", "en")
    assert code == 0 and stdout.strip() == "cli_voice"
    code, _, _ = run(capsys, "convert", str(ref_wav), "--voice", "cli_voice", "-o", str(out))
    assert code == 0
    code, stdout, _ = run(capsys, "voices", "list", "--kind", "cloned", "--json")
    assert [v["id"] for v in json.loads(stdout)] == ["cli_voice"]
    code, _, _ = run(capsys, "voices", "remove", "cli_voice")
    assert code == 0


def test_stdin_and_errors(engine, tmp_path, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    out = tmp_path / "s.wav"
    code, _, _ = run(capsys, "turbo", "-o", str(out))
    assert code == 0 and out.is_file()
    code, _, err = run(capsys, "turbo", "Hi", "--voice", "nobody", "-o", str(out))
    assert code == 2 and "Unknown voice" in err


def test_voices_table_and_watermark(engine, ref_wav, capsys):
    code, stdout, _ = run(capsys, "voices", "list")
    assert code == 0 and "sample_zh" in stdout and "Chinese" in stdout
    code, stdout, _ = run(capsys, "watermark", str(ref_wav))
    assert code == 0 and "no watermark" in stdout


def test_prompts_save_list_show_and_use(engine, tmp_path, capsys):
    code, stdout, err = run(capsys, "prompts", "save", "French Hello", "Bonjour [settings cfg_weight=0]à tous[/settings]",
                            "--mode", "multilingual", "-l", "fr", "--voice", "podcast", "--temperature", "0.6")
    assert code == 0 and stdout.strip() == "french_hello", err
    code, stdout, _ = run(capsys, "prompts", "list")
    assert "french_hello" in stdout and "multilingual" in stdout and "podcast" in stdout
    code, stdout, _ = run(capsys, "prompts", "show", "french_hello")
    assert stdout.strip() == "Bonjour [settings cfg_weight=0]à tous[/settings]"
    out = tmp_path / "p.wav"
    code, stdout, err = run(capsys, "multilingual", "--prompt", "french_hello", "-o", str(out), "--json")
    assert code == 0, err
    meta = json.loads(stdout)
    assert meta["language"] == "fr" and meta["voice"] == "podcast"
    call = [c for c in engine.backend.calls if c[0] == "synthesize"][-1][1]
    assert call["params"].temperature == 0.6
    # Explicit arguments win over the prompt.
    code, stdout, _ = run(capsys, "multilingual", "Hola", "--prompt", "french_hello", "-l", "es", "--voice", "default",
                          "--temperature", "0.9", "-o", str(out), "--json")
    assert json.loads(stdout)["language"] == "es" and json.loads(stdout)["voice"] == "default"
    assert [c for c in engine.backend.calls if c[0] == "synthesize"][-1][1]["params"].temperature == 0.9
    code, _, _ = run(capsys, "prompts", "remove", "french_hello")
    assert code == 0
    code, _, err = run(capsys, "turbo", "--prompt", "french_hello", "-o", str(out))
    assert code == 2 and "Unknown prompt" in err


def test_transcribe(engine, ref_wav, capsys):
    code, stdout, err = run(capsys, "transcribe", str(ref_wav), "-l", "en")
    assert code == 0 and stdout.startswith("Placeholder transcript"), err
    code, stdout, _ = run(capsys, "transcribe", str(ref_wav), "--json")
    assert json.loads(stdout)["segments"]


def test_emotion_presets(engine, tmp_path, capsys):
    out = tmp_path / "e.wav"
    code, stdout, err = run(capsys, "original", "Hi", "--emotion", "angry", "--temperature", "0.5", "-o", str(out), "--json")
    assert code == 0, err
    assert json.loads(stdout)["preset"] == "angry"
    sent = [c for c in engine.backend.calls if c[0] == "synthesize"][-1][1]["params"]
    assert (sent.exaggeration, sent.temperature) == (1.1, 0.5)
    code, stdout, _ = run(capsys, "presets")
    assert code == 0 and "happy" in stdout and "exaggeration=" in stdout
    code, stdout, _ = run(capsys, "presets", "--json")
    assert {p["id"] for p in json.loads(stdout)} >= {"sad", "excited"}
    with pytest.raises(SystemExit):
        main(["--backend", "mock", "turbo", "Hi", "--emotion", "giddy"])
