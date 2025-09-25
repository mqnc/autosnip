# autosnip
Automatically clean up voice recordings

You need a text file containing the script and a wav file containing your recording of the script. When you make a mistake, you say a cue word and start over at an earlier point. Then autosnip stitches the good parts together and spits out a ready-to-use wav file.

## Setup

Install autosnip:
```bash
pip install autosnip
```

Download and unpack a speech recognition model from https://alphacephei.com/vosk/models for your language.

## Usage

Carefully craft your script with depth and wit. Think of a cue word to signal a mistake to autosnip. The cue word should be a word that does not occur in your script. Record yourself reading the script. Whenever you make a mistake or want to rerecord some part for whatever reason, say the cue word and start over at some earlier point in the script, a good place is the start of the current sentence so there is no break in the flow mid sentence. When you are done, feed the script and your recording into autosnip:

```
autosnip -s my_script.txt -r my_recording.wav -o my_clean_recording.wav -m vosk-model-en-us-0.22 -c oops -v
```

Congratulations! You may now publish your recording into the world.

## Example

Check out the example directory.

## How it Works

Your recording is transcribed using a VOSK speech recognition model. Start and end time of each word are stored. The transcript is then split apart at each occurance of the cue word. For each section, the whole script is scanned to see where this section fits best, so in theory you don't have to read the parts in order. The sections are patched into the script in the order you recorded them, always overwriting any previous recording. Then the corresponding intervals are selected from the recorded wav file and stitched together with short cross fades.

If you are unhappy with the result at some place, you can just make a new recording of that part (starting with the cue word), append it to your previous recording and rerun autosnip.

## Project Status

The happy path is working, however the project is still young and not battle-tested. Feel free to report issues! There is also not yet much performance optimization going on. The runtime is probably O(n²) where n is the number of words in your script, so it may be a good idea to record chapter by chapter instead of a whole book in one go.

### Happy Recording!