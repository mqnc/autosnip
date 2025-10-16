import argparse
import hashlib
import json
import os
import re
import sys
import wave
from colorsys import hls_to_rgb
from functools import cache
from math import cos
from shutil import get_terminal_size
from types import SimpleNamespace
from typing import Dict, List, Tuple

import soundfile as sf
import numpy as np
from vosk import KaldiRecognizer, Model

if sys.version_info < (3, 13):
    import audioop
else:
    import audioop_lts as audioop

import tempfile

VERBOSE = False  # will be set by -v flag in setup()


def v_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def cleanup(text, remove_spaces=False):
    # lowercase, remove punctuation, trim whitespace
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # replace all non-letters with space
    text = re.sub(
        r"\s+", "" if remove_spaces else " ", text
    )  # remove or merge all whitespaces
    text = text.strip()
    return text


ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"
ANSI_HIGHLIGHT = "\033[45m"
ANSI_DIM = "\033[90m"


def ansi_color(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


@cache
def ansi_fib_color(i) -> str:
    # generate visually distinct colors by index
    phi = (1 + 5**0.5) / 2
    hue = (i * phi + 0.7) % 1
    lightness = 0.5 + 0.2 * cos(i * 3)
    r, g, b = [int(255 * c) for c in hls_to_rgb(hue, lightness, 1)]
    return ansi_color(r, g, b)


def setup():

    # fmt: off
    parser = argparse.ArgumentParser(
        description="autosnip: automatically clean up voice recordings based on a script and a cue word",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", "--script", default="script.txt", help="path to the script file")
    parser.add_argument("-r", "--rec", default="recording.wav", help="path to the input WAV recording")
    parser.add_argument("-o", "--out", default="recording_cleaned.wav", help="output WAV file name")
    parser.add_argument("-m", "--model", default="vosk-model-*", help="Vosk model directory")
    parser.add_argument("-c", "--cue", default="correction", help="cue word to start over at an earlier point in the script")
    parser.add_argument("-w", "--words", type=int, default="1", help="minimum number of words under which a segment is discarded")
    parser.add_argument("-x", "--exclude", nargs='*', type=int, help="manually exclude these segments (by index)")
    parser.add_argument("-f", "--fade", type=float, default=0.01, help="crossfade duration in seconds")
    parser.add_argument("-C", "--cache", default=tempfile.gettempdir(), help="cache folder for the down-sampled audio and the transcript")
    parser.add_argument("-W", "--wavs", help="folder to output wav snippets to make later manual corrections easier")
    parser.add_argument("-q", "--quiet", action="store_true", help="no informative and very pretty output")
    args = parser.parse_args()
    # fmt: on

    global VERBOSE
    VERBOSE = not args.quiet

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.wavs, exist_ok=True)

    if (
        args.script == parser.get_default("script")
        and args.rec == parser.get_default("rec")
        and (not os.path.exists(args.rec) or not os.path.exists(args.script))
    ):
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.script):
        print(f"could not find script file {args.script}")
        sys.exit(1)

    if not os.path.exists(args.rec):
        print(f"could not find recording file {args.rec}")
        sys.exit(1)

    if args.model == parser.get_default("model"):
        models = [
            d
            for d in os.listdir(".")
            if os.path.isdir(d) and d.startswith("vosk-model-")
        ]
        if not models:
            print("could not find a vosk-model-* folder in the current directory;")
            print(
                "please download a Vosk model from https://alphacephei.com/vosk/models"
            )
            print("and unpack it here or specify the path using -m/--model")
            sys.exit(1)
        elif len(models) > 1:
            print(f"found multiple Vosk model candidates: {', '.join(models)};")
            print("please select one using -m/--model")
            sys.exit(1)
        model_path = models[0]
    else:
        model_path = args.model

    cue = cleanup(args.cue)
    if " " in cue or not cue:
        print("cue word must be a single word")
        sys.exit(1)

    return SimpleNamespace(
        script=args.script,
        rec=args.rec,
        out=args.out,
        model_path=model_path,
        cue=cue,
        segment_min_words=args.words,
        manually_excluded_segments=[] if args.exclude is None else args.exclude,
        fade=args.fade,
        cache=args.cache,
        wavdir=args.wavs,
    )


def transcribe_audio(model, wav_path):
    wf = wave.open(wav_path, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    result = []
    v_print(f"\n{ANSI_BOLD}transcribing...{ANSI_RESET}\n")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            heard = json.loads(rec.Result())
            if "result" in heard:
                v_print(
                    " ".join([w["word"] for w in heard["result"]]), end=" ", flush=True
                )
            result.append(heard)
    heard = json.loads(rec.FinalResult())
    if "result" in heard:
        v_print(" ".join([w["word"] for w in heard["result"]]), end="", flush=True)
    v_print()
    result.append(heard)
    words = []
    for r in result:
        if "result" in r:
            words.extend(r["result"])

    return words


def load_or_generate_transcript(cfg):

    with open(cfg.rec, "rb") as f:
        audio_checksum = hashlib.md5(f.read()).hexdigest()

    with open(cfg.script, "r") as f:
        script = cleanup(f.read())

    model_name = os.path.basename(os.path.normpath(cfg.model_path))

    combined_hash = hashlib.md5(
        (audio_checksum + model_name).encode("utf-8")
    ).hexdigest()
    transcript_cache = os.path.join(cfg.cache, f"{combined_hash}.json")

    if os.path.exists(transcript_cache):
        print(f"using cached transcript {transcript_cache}")
        with open(transcript_cache, "r") as f:
            transcript = json.load(f)
        print(" ".join([ts["word"] for ts in transcript]))
    else:
        cache_16k = os.path.join(cfg.cache, f"{audio_checksum}.16k.wav")
        if os.path.exists(cache_16k):
            print(f"using cached down-sampled audio {cache_16k}")
        else:
            data, sr_in = sf.read(cfg.rec)
            sr_out = 16000
            ratio = sr_out / sr_in
            x_old = np.arange(len(data))
            x_new = np.linspace(0, len(data) - 1, int(len(data) * ratio))
            data_resampled = np.interp(x_new, x_old, data)
            sf.write(cache_16k, data_resampled, sr_out)

            # data, samplerate = sf.read(cfg.rec)
            # sf.write(cache_16k, data, 16000)
            print(f"down-sampled audio written to {cache_16k}")

        model = Model(cfg.model_path)
        if model.vosk_model_find_word(cfg.cue) == -1:
            print(
                f"{ANSI_YELLOW}warning: cue word '{cfg.cue}' not found in the model vocabulary{ANSI_RESET}"
            )

        transcript = transcribe_audio(model, cache_16k)
        with open(transcript_cache, "w") as f:
            json.dump(transcript, f)
            print(f"transcript cached to {transcript_cache}")

    for w in transcript:
        w["word"] = cleanup(w["word"], remove_spaces=True)

    return script.split(" "), transcript


def split_into_segments(transcript, cue, segment_min_words, manually_excluded_segments):
    segments = []
    current_seg = []
    i = 0
    for w in transcript:
        w["seg"] = i
        if w["word"] == cue:
            i += 1
            if current_seg:
                segments.append(current_seg)
                current_seg = []
            continue
        current_seg.append(w)
    if current_seg:
        segments.append(current_seg)

    if VERBOSE:
        print(
            f"\n{ANSI_BOLD}contiguous segments (separated by cue word '{cue}'):{ANSI_RESET}\n"
        )
        for i, s in enumerate(segments):
            if len(s) >= segment_min_words and i not in manually_excluded_segments:
                color = ansi_fib_color(i)
            else:
                color = ANSI_HIGHLIGHT
            print(
                color, f"({i}) ", " ".join([w["word"] for w in s]), ANSI_RESET, sep=""
            )

    return segments


@cache
def levenshtein_words(s1, s2):
    # compute the Levenshtein distance between two strings to measure how different they are
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[len(s1)][len(s2)]


def levenshtein_word_sequences(
    source: List[str], segment: List[Dict]
) -> Tuple[int, List[str]]:
    # compute the sort-of Levenshtein distance between two sequences of words:
    # insertion and deletion cost is the length of the word
    # (deleting words at the beginning or end is free)
    # substitution cost is the Levenshtein distance between the two words

    # source is a list of strings (written-only script)
    # segment is kaldi transcript segment (list of dicts with 'word', 'start', 'end')

    # create a distance matrix
    dp = [[(0, None)] * (len(segment) + 1) for _ in range(len(source) + 1)]
    # matrix entry: (distance, operation)

    for i in range(1, len(source) + 1):
        # deletions in the beginning are free
        dp[i][0] = (0, "d")
    for j in range(1, len(segment) + 1):
        dp[0][j] = (dp[0][j - 1][0] + len(segment[j - 1]["word"]), "i")

    # compute the Levenshtein distance
    for i in range(1, len(source) + 1):
        for j in range(1, len(segment) + 1):

            if j == len(segment):
                cost_delete_word = 0  # deletions at the end are free
            else:
                cost_delete_word = len(source[i - 1])
            cost_insert_word = len(segment[j - 1]["word"])
            cost_substitute_word = levenshtein_words(
                source[i - 1], segment[j - 1]["word"]
            )

            cost_deletion = dp[i - 1][j][0] + cost_delete_word
            cost_insertion = dp[i][j - 1][0] + cost_insert_word
            cost_substitution = dp[i - 1][j - 1][0] + cost_substitute_word

            if cost_deletion <= cost_insertion and cost_deletion <= cost_substitution:
                dp[i][j] = (cost_deletion, "d")
            elif cost_insertion <= cost_substitution:
                dp[i][j] = (cost_insertion, "i")
            else:
                dp[i][j] = (cost_substitution, "s")

    # Reconstruct the sequence of operations
    reverse_edits = []
    correspondences = [None] * len(source)
    i = len(source)
    j = len(segment)
    while (i, j) != (0, 0):
        if dp[i][j][1] == "d":
            reverse_edits.append("d")
            i -= 1
        elif dp[i][j][1] == "i":
            reverse_edits.append("i")
            j -= 1
        else:
            reverse_edits.append("s")
            correspondences[i - 1] = j - 1
            i -= 1
            j -= 1
    edits = "".join(reversed(reverse_edits))

    # find first and last matched word in the source
    first = next((i for i, val in enumerate(correspondences) if val is not None), None)
    reverse_corresps = reversed(list(enumerate(correspondences)))
    last = next((i for i, val in reverse_corresps if val is not None), None)

    return first, last, correspondences, edits


def show_difference(
    s1: List[str], s2: List[dict], edit_sequence: str, intro_outro=True
):
    if not VERBOSE:
        return
    i, j = 0, 0
    max_len = get_terminal_size().columns
    script_line, script_line_len = "", 0
    voice_line, voice_line_len = "", 0

    def append(
        script_word, voice_word, script_color="", voice_color="", highlight_diff=True
    ):
        nonlocal script_line, script_line_len, voice_line, voice_line_len
        length = 1 + max(len(script_word), len(voice_word))
        if script_line_len + length > max_len:
            print(script_line)
            print(voice_line)
            script_line, script_line_len = "", 0
            voice_line, voice_line_len = "", 0
        if highlight_diff and script_word != voice_word:
            script_line += ANSI_HIGHLIGHT
            voice_line += ANSI_HIGHLIGHT
        else:
            script_line += script_color
            voice_line += voice_color

        script_line += (
            script_word + " " * (length - len(script_word) - 1) + ANSI_RESET + " "
        )
        voice_line += (
            voice_word + " " * (length - len(voice_word) - 1) + ANSI_RESET + " "
        )
        script_line_len += length
        voice_line_len += length

    started = False
    i_seg_before = None
    for op in edit_sequence.rstrip("d"):
        if op == "s":
            if not started and intro_outro:
                for i_ in range(max(i - 3, 0), i):
                    append(s1[i_], "", script_color=ANSI_DIM, highlight_diff=False)
            started = True
            i_seg = s2[j]["seg"]
            if i_seg_before != i_seg:
                i_seg_before = i_seg
                append(
                    "",
                    f"({i_seg})",
                    voice_color=ansi_fib_color(i_seg),
                    highlight_diff=False,
                )
            append(s1[i], s2[j]["word"], voice_color=ansi_fib_color(i_seg))
            i += 1
            j += 1
        elif op == "d":
            if started:
                append(s1[i], "")
            i += 1
        elif op == "i":
            if not started and intro_outro:
                for i_ in range(max(i - 3, 0), i):
                    append(s1[i_], "", highlight_diff=False)
            started = True
            i_seg = s2[j]["seg"]
            if i_seg_before != i_seg:
                i_seg_before = i_seg
                append(
                    "",
                    f"({i_seg})",
                    voice_color=ansi_fib_color(i_seg),
                    highlight_diff=False,
                )
            append("", s2[j]["word"], voice_color=ansi_fib_color(s2[j]["seg"]))
            j += 1

    for i_ in range(i, min(i + 3, len(s1))):
        append(s1[i_], "", script_color=ANSI_DIM, highlight_diff=False)

    print(script_line)
    print(voice_line)
    print()


def find_correspondences(script, segments, discarded_segments):
    # for each word in the script, find the corresponding segment and word index

    script_corresps = [(None, None)] * len(script)

    v_print(f"\n{ANSI_BOLD}correspondences:{ANSI_RESET}\n")

    for i, segment in enumerate(segments):
        if i in discarded_segments:
            continue
        first, last, seg_corresps, edits = levenshtein_word_sequences(script, segment)
        show_difference(script, segment, edits)

        # overwrite previous script correspondences in case of overlap
        for k in range(first, last + 1):
            script_corresps[k] = (i, seg_corresps[k])

    return script_corresps


def select_clips(script, segments, script_corresps):
    # find contiguous chunks that correspond to the same segment

    # v_print(
    #     f"{ANSI_BOLD}final composition of selected clips, missing script parts highlighted:{ANSI_RESET}\n"
    # )

    intervals = []
    composed_script = []

    i_script_word = 0
    while True:

        i_seg, i_first_part = script_corresps[i_script_word]

        if i_seg == None or i_first_part is None:
            i_first_script_word = i_script_word
            while (
                i_script_word + 1 < len(script_corresps)
                and script_corresps[i_script_word + 1][0] == None
            ):
                i_script_word += 1

            # if VERBOSE:
            #     print(ANSI_HIGHLIGHT, end="")
            #     print(" ".join(script[i_first_script_word : i_script_word + 1]), end="")
            #     print(ANSI_RESET + " ", end="")
        else:
            segment = segments[i_seg]
            clip_start_time = segment[i_first_part]["start"]
            i_last_part = i_first_part

            while (
                i_script_word + 1 < len(script_corresps)
                and script_corresps[i_script_word + 1][0] == i_seg
            ):
                i_script_word += 1
                _, test = script_corresps[i_script_word]
                if test is not None:
                    i_last_part = test

            if i_last_part + 1 < len(segment):
                clip_end_time = segments[i_seg][i_last_part + 1]["start"]
            else:
                clip_end_time = segments[i_seg][i_last_part]["end"]

            intervals.append((clip_start_time, clip_end_time))
            composed_script += segment[i_first_part : i_last_part + 1]

            # if VERBOSE:
            #     print(f"{ansi_fib_color(i_seg)}({i_seg})", end=" ")
            #     chunk = segment[i_first_part : i_last_part + 1]
            #     print(" ".join([p["word"] for p in chunk]), end="")
            #     print(ANSI_RESET + " ", end="")

        i_script_word += 1
        if i_script_word >= len(script):
            break

    v_print("\n")

    return intervals, composed_script


def export_audio_segments(recording_path, segments, output_dir):

    with wave.open(recording_path, "rb") as in_wav:
        params = in_wav.getparams()
        frame_rate = in_wav.getframerate()
        digits = len(str(len(segments)))
        for i, segment in enumerate(segments):
            start = segment[0]["start"]
            end = segment[-1]["end"]
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            in_wav.setpos(start_frame)
            wav_segment = in_wav.readframes(end_frame - start_frame)
            title = "_".join([w["word"] for w in segment[0 : min(len(segment), 3)]])
            out_path = os.path.join(output_dir, f"{i:0{digits}d}_{title}.wav")
            with wave.open(out_path, "wb") as out_wav:
                out_wav.setparams(params)
                out_wav.writeframes(wav_segment)


def stitch_audio(recording_path, intervals, output_path, crossfade):

    with wave.open(recording_path, "rb") as in_wav:
        params = in_wav.getparams()
        frame_rate = in_wav.getframerate()
        sample_width = in_wav.getsampwidth()
        num_channels = in_wav.getnchannels()
        frame_size = sample_width * num_channels

        with wave.open(output_path, "wb") as out_wav:
            out_wav.setparams(params)
            fade_samples = int(crossfade * frame_rate)
            prev_segment = (
                b"\x00" * fade_samples * frame_size
            )  # fade in from silence at the beginning
            for start, end in intervals + [(None, None)]:
                if start is not None:
                    start_frame = int(start * frame_rate)
                    end_frame = int(end * frame_rate)
                    in_wav.setpos(start_frame)
                    segment = in_wav.readframes(end_frame - start_frame)
                    fade_in = segment[: fade_samples * frame_size]
                else:
                    # fade out to silence at the end
                    fade_in = b"\x00" * fade_samples * frame_size
                fade_out = prev_segment[-fade_samples * frame_size :]
                mixed = bytearray()
                for i in range(fade_samples):
                    a = audioop.getsample(fade_out, sample_width, i)
                    b = audioop.getsample(fade_in, sample_width, i)
                    t = i / fade_samples
                    gain_a = (1 - t) ** 0.5
                    gain_b = t**0.5
                    mixed_sample = int(a * gain_a + b * gain_b)
                    mixed.extend(
                        audioop.lin2lin(
                            mixed_sample.to_bytes(sample_width, "little", signed=True),
                            sample_width,
                            sample_width,
                        )
                    )
                # write crossfade section
                out_wav.writeframes(mixed)

                if start is None:
                    break

                # write current section until next crossfade section begins
                out_wav.writeframes(
                    segment[fade_samples * frame_size : -fade_samples * frame_size]
                )
                prev_segment = segment


def main():
    cfg = setup()

    script, transcript = load_or_generate_transcript(cfg)

    if cfg.cue not in [w["word"] for w in transcript]:
        print(
            f"{ANSI_YELLOW}warning: cue word '{cfg.cue}' not found in the transcript,"
        )
        print("just copying input to output{ANSI_RESET}")
        with open(cfg.out, "wb") as out_f, open(cfg.rec, "rb") as in_f:
            out_f.write(in_f.read())
        sys.exit(0)

    segments = split_into_segments(
        transcript, cfg.cue, cfg.segment_min_words, cfg.manually_excluded_segments
    )

    if cfg.wavdir is not None:
        export_audio_segments(cfg.rec, segments, cfg.wavdir)

    discarded_segments = cfg.manually_excluded_segments + [
        i for i, s in enumerate(segments) if len(s) < cfg.segment_min_words
    ]

    correspondences = find_correspondences(script, segments, discarded_segments)
    # with open("corr.json", "w") as f:
    #     json.dump(correspondences, f)
    # with open("corr.json", "r") as f:
    #     correspondences = json.load(f)

    intervals, composed_script = select_clips(script, segments, correspondences)

    if VERBOSE:
        print(
            f"{ANSI_BOLD}final composition of selected clips (might take some time):\n"
        )
        first, last, seg_corresps, edits = levenshtein_word_sequences(
            script, composed_script
        )
        show_difference(script, composed_script, edits, intro_outro=False)

    v_print(f"{ANSI_BOLD}audio segments:{ANSI_RESET}\n")
    for start, end in intervals:
        v_print(f"{start:.2f} - {end:.2f} ({end - start:.2f} s)")

    stitch_audio(cfg.rec, intervals, cfg.out, cfg.fade)

    v_print(f"\ncleaned recording written to {cfg.out}")


if __name__ == "__main__":
    main()
