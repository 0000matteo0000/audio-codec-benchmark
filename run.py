#!/usr/bin/env python3
import glob
import logging

from utils import (generate_white_sound, load_checkpoint, plot_merged_results,
                   run_benchmark, save_table)

# setup logging
logging.basicConfig(format="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style="{", level="WARNING", encoding="utf-8", errors="surrogateescape",)

# files to use for benchmarks
files = [
    *glob.iglob(glob.escape("./test_files/") + "*"),
]

# useless "white" noise generator
if False:
    if False:
        generate_white_sound()
    files.append("./white.wav")

codecs = {
    "aac": {
        "32k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "32k", "profile": "aac_ltp", },
        "40k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "40k", "profile": "aac_ltp", },
        "48k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "48k", "profile": "aac_ltp", },
        "56k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "56k", "profile": "aac_ltp", },
        "64k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "64k", "profile": "aac_ltp", },
        "80k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "80k", "profile": "aac_ltp", },
        "96k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "96k", "profile": "aac_ltp", },
        "112k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "112k", "profile": "aac_ltp", },
        "128k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "128k", "profile": "aac_ltp", },
        "144k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "144k", "profile": "aac_ltp", },
        "160k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "160k", "profile": "aac_ltp", },
        "192k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "192k", "profile": "aac_ltp", },
        "224k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "224k", "profile": "aac_ltp", },
        "256k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "256k", "profile": "aac_ltp", },
        "320k": {"format": "matroska", "acodec": "aac", "ar": "48k", "audio_bitrate": "320k", "profile": "aac_ltp", },
    },
    "ac3": {
        "32k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "32k", },
        "40k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "40k", },
        "48k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "48k", },
        "56k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "56k", },
        "64k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "64k", },
        "80k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "80k", },
        "96k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "96k", },
        "112k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "112k", },
        "128k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "128k", },
        "144k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "144k", },
        "160k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "160k", },
        "192k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "192k", },
        "224k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "224k", },
        "256k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "256k", },
        "320k": {"format": "matroska", "acodec": "ac3", "ar": "48k", "audio_bitrate": "320k", },
    },
    "mp3": {
        "32k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "32k", "abr": "1", "compression_level": "0", },
        "40k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "40k", "abr": "1", "compression_level": "0", },
        "48k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "48k", "abr": "1", "compression_level": "0", },
        "56k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "56k", "abr": "1", "compression_level": "0", },
        "64k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "64k", "abr": "1", "compression_level": "0", },
        "80k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "80k", "abr": "1", "compression_level": "0", },
        "96k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "96k", "abr": "1", "compression_level": "0", },
        "112k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "112k", "abr": "1", "compression_level": "0", },
        "128k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "128k", "abr": "1", "compression_level": "0", },
        "144k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "144k", "abr": "1", "compression_level": "0", },
        "160k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "160k", "abr": "1", "compression_level": "0", },
        "192k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "192k", "abr": "1", "compression_level": "0", },
        "224k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "224k", "abr": "1", "compression_level": "0", },
        "256k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "256k", "abr": "1", "compression_level": "0", },
        "320k": {"format": "matroska", "acodec": "libmp3lame", "ar": "48k", "audio_bitrate": "320k", "abr": "1", "compression_level": "0", },
    },
    "opus": {
        "32k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "32k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "40k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "40k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "48k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "48k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "56k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "56k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "64k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "64k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "80k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "80k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "96k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "96k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "112k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "112k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "128k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "128k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "144k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "144k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "160k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "160k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "192k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "192k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "224k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "224k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "256k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "256k", "vbr": "on", "compression_level": "10", "application": "audio", },
        "320k": {"format": "matroska", "acodec": "libopus", "ar": "48k", "audio_bitrate": "320k", "vbr": "on", "compression_level": "10", "application": "audio", },
    },
}


if __name__ == "__main__":
    if True:
        merged_results = run_benchmark(files, codecs, resume=True)
    else:
        info, results, merged_results = load_checkpoint()
    save_table(merged_results)
    plot_merged_results(merged_results)
    logging.warning("---------- Done. ----------")
