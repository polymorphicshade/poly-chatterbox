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
