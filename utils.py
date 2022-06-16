import errno
import json
import logging
import math
import os
import platform
import re
import signal
import traceback
from hashlib import sha3_384
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import ffmpeg
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import coherence, correlate, correlation_lags, spectrogram


def get_info(path):
    info = ffmpeg.probe(path)
    # print(json.dumps(info, indent="\t"))
    return info


def decode(path, path_out, **ffmpeg_args):
    """
    ffmpeg_args should -map only what should be evaluated
    the output of this function wil
    l be encoded as a temporary file
    """
    out, _ = (
        ffmpeg
        .input(path)
        .output(path_out, **ffmpeg_args, loglevel="warning")
        .overwrite_output()
        .run()
    )
    return out


def decode_audio_wav_pcm_s32le(path, path_out, **ffmpeg_args):
    return decode(path, path_out, format="wav", acodec="pcm_s32le", **ffmpeg_args)


def load_audio_wav_pcm_s32le(path):
    return wavfile.read(path)


def save_audio_wav_pcm_s32le(data, samplerate, path_out):
    return wavfile.write(path_out, samplerate, data)


def encode(path, path_out="tmp", **ffmpeg_args):
    out, _ = (
        ffmpeg
        .input(path)
        .output(path_out, **ffmpeg_args, loglevel="warning")
        .overwrite_output()
        .run()
    )
    return out


def get_trim_parameters(original_name, samples, sample_lenght=1):
    times = np.linspace(0, float(get_info(original_name)["format"]["duration"]), num=samples + 2, endpoint=True)[1:-1]
    trims = [f"[a]atrim=start={round(t,2)}:duration={sample_lenght}[{i}]" for i, t in enumerate(times)]
    trims.append("".join(f"[{i}]" for i in range(len(times))) + f"concat=n={len(times)}:v=0:a=1[out]")
    return {"filter_complex": ";".join(trims), "map": ["out"]}


def encode_decode_load(original_name, samplerate=None, parameters=None, encoded_name="encoded", decoded_name="decoded.wav", samples=None, sample_lenght=1):
    if parameters is not None:
        extra_args = get_trim_parameters(original_name, samples, sample_lenght=sample_lenght) if samples is not None else {}
        logging.info(f">> Converting {original_name!r} with parameters: {parameters} ...")
        t_start = perf_counter()
        encode(original_name, path_out=encoded_name, **parameters, **extra_args)
        t_end = perf_counter()
        extra_args = {}  # do not trim again when decoding
        encoding_time = (t_end - t_start)
    else:
        extra_args = get_trim_parameters(encoded_name, samples, sample_lenght=sample_lenght) if samples is not None else {}
        logging.info(f">> Skipping conversion for {original_name!r}")
        encoding_time = 0
    logging.info(f">> Deconding {encoded_name!r} ...")
    if samplerate is not None:
        t_start = perf_counter()
        decode_audio_wav_pcm_s32le(encoded_name, path_out=decoded_name, ar=samplerate, **extra_args)
        t_end = perf_counter()
    else:
        t_start = perf_counter()
        decode_audio_wav_pcm_s32le(encoded_name, path_out=decoded_name, **extra_args)
        t_end = perf_counter()
    decoding_time = (t_end - t_start)
    logging.info(f">> Loading {decoded_name!r} ...")
    decoded_samplerate, decoded = load_audio_wav_pcm_s32le(decoded_name)
    return decoded_samplerate, decoded, encoding_time, decoding_time


def align(a, b, limit=None):  # useless, worsens signal coherence
    la, lb = a[:limit], b[:limit]
    if len(a.shape) == 1:
        correlation = correlate(la, lb)
        lag = correlation_lags(la.size, lb.size)[np.argmax(correlation)]
        lag = mean(lag)
        if lag > 0:
            a = np.vstack((np.zeros(abs(lag)), a))
            b = np.vstack((b, np.zeros(abs(lag))))
        else:
            a = np.vstack((a, np.zeros(abs(lag))))
            b = np.vstack((np.zeros(abs(lag)), b))
    else:
        lag = []
        for channel in range(a.shape[1]):
            correlation = correlate(la[channel], lb[channel])
            lag.append(correlation_lags(la.size, lb.size)[np.argmax(correlation)])
        lag = mean(lag)
        if lag > 0:
            a = np.vstack((np.zeros((abs(lag), a.shape[1])), a))
            b = np.vstack((b, np.zeros((abs(lag), a.shape[1]))))
        else:
            a = np.vstack((a, np.zeros((abs(lag), a.shape[1]))))
            b = np.vstack((np.zeros((abs(lag), a.shape[1])), b))
    return a, b


def generate_white_sound(frequency_range=None, length=1, samplerate=96000, dtype=np.int32):
    # --------------------
    # for dtype in [
    #     np.int8,
    #     np.uint8,
    #     np.int16,
    #     np.uint16,
    #     np.int32,
    #     np.uint32,
    #     np.int64,
    #     int
    #     np.uint64,
    #     np.float16,
    #     np.float32,
    #     np.float64,
    #     float
    # ]:
    # --------------------
    try:
        info = np.iinfo(dtype)
        ptype = int
        center = (info.max + info.min) // 2 + 1
        amplitude = info.max - center
    except ValueError:
        info = np.finfo(dtype)
        ptype = float
        center = (info.max + info.min) / 2
        amplitude = info.max - center
    # --------------------
        # print(dtype)
        # print("ca", center, amplitude)
        # print("mm", info.min, info.max)
        # print("ul", center - amplitude, center + amplitude)
    # --------------------
    # use python instead of numpy to avoid overflows
    if frequency_range is None:
        n = 4096
        fmi = 12
        fma = 24000
        lfmi = math.log10(fmi)
        lfma = math.log10(fma)
        lfs = (lfma - lfmi) / (n - 1)
        frequency_range = []
        lf = lfmi
        while lf <= lfma:
            frequency_range.append(math.pow(10, lf))
            lf += lfs
        # frequency_range.append(math.pow(10, lfma))
    times = range(samplerate * length)
    n = len(frequency_range)
    logging.debug(n)
    samples = [0 for t in times]
    for i, frequency in enumerate(frequency_range):
        for t in times:
            samples[t] += center + amplitude * math.sin(2 * math.pi / float(frequency) * t)
        logging.debug(i, frequency)
    for t in times:
        if ptype is int:
            samples[t] = samples[t] // n
        if ptype is float:
            samples[t] = samples[t] / n
    samples = np.asarray(samples, dtype=dtype)
    wavfile.write("white.wav", samplerate, samples)


def channel_power_spectro(audio, samplerate, window_duration_seconds=None, min_frequency=12, max_frequency=24000):
    if window_duration_seconds is None:
        window_duration_seconds = 1 / min_frequency
    samples_in_full_windows = int(audio.shape[0] / samplerate / window_duration_seconds) * samplerate
    nperseg = int(samplerate * window_duration_seconds)
    f, t, sxx = spectrogram(audio[:samples_in_full_windows], fs=samplerate, nperseg=nperseg, nfft=max(nperseg, int(max_frequency - min_frequency)), mode="magnitude")
    return f, t, sxx**2  # sxx**2 converts amplitude spectreo to power spectro and will always be positive


def power_spectro(audio, samplerate, window_duration_seconds=None, min_frequency=12, max_frequency=24000):
    c = []
    i = None
    if len(audio.shape) == 1:
        f, t, sxx = channel_power_spectro(audio, samplerate, window_duration_seconds=window_duration_seconds, min_frequency=min_frequency, max_frequency=max_frequency)
        if i is None:
            # all channels are expected of same len
            i = np.where((f >= min_frequency) & (f <= max_frequency))
        c.append(sxx[i])  # ignore inhaudible frequencies and convert to db
    else:
        for channel in range(audio.shape[1]):
            f, t, sxx = channel_power_spectro(audio[:, channel], samplerate, window_duration_seconds=window_duration_seconds, min_frequency=min_frequency, max_frequency=max_frequency)
            if i is None:
                # all channels are expected of same len
                i = np.where((f >= min_frequency) & (f <= max_frequency))
            c.append(sxx[i])  # ignore inhaudible frequencies and convert to db
    return f[i], t, np.asarray(c)


def power_spectro_to_db(power_spectro):
    power_spectro[power_spectro == 0] = np.finfo(power_spectro.dtype).resolution
    return 10 * np.log10(power_spectro)


def signal_coherence_e(clear_signal, noisy_signal, samplerate, window_duration_seconds=None, min_frequency=12, max_frequency=24000):
    if window_duration_seconds is None:
        window_duration_seconds = 1 / min_frequency
    nperseg = int(samplerate * window_duration_seconds)
    f, c = coherence(
        clear_signal, noisy_signal,
        fs=samplerate,
        nperseg=nperseg,
        # nfft=max(nperseg, int(max_frequency - min_frequency)),
        axis=0,
    )
    i = np.where((f >= min_frequency) & (f <= max_frequency))
    return f[i], c[i]


def signal_incoherence(signal_coherence, q=None):
    if q is None:
        return 1 - np.mean(signal_coherence).item()
    else:
        return 1 - np.percentile(signal_coherence, q=q).item()


def array_hash(audio):
    return sha3_384(audio).digest().hex()


def a_weights_db(frequencies):
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    return 2.0 + 20.0 * (
        np.log10(const[0]) +
        +2 * np.log10(f_sq) +
        -np.log10(f_sq + const[0]) +
        -np.log10(f_sq + const[1]) +
        -0.5 * np.log10(f_sq + const[2]) +
        -0.5 * np.log10(f_sq + const[3])
    )


f = np.asarray([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])
e_f = np.hstack((10, 12.5, 16, f, 16000, 20000, 25000, 31500))
af = np.asarray([0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301])
e_af = np.hstack((af[0], af[0], af[0], af, af[-2], af[3], af[0], af[0]))
Lu = np.asarray([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1])
e_Lu = np.hstack((Lu[0], Lu[0], Lu[0], Lu, Lu[-2], Lu[3], Lu[0], Lu[0]))
Tf = np.asarray([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])
e_Tf = np.hstack((Tf[0], Tf[0], Tf[0], Tf, Tf[-2], Tf[3], Tf[0], Tf[0]))


def iso226_spl_contour(Ln=40):
    if Ln < 0 or Ln > 90:
        raise AssertionError("Ln out of bounds [0, 90]")
    Af = 4.47e-3 * (10.0**(0.025 * Ln) - 1.15) + (0.4 * 10.0**((Tf + Lu) / 10.0 - 9.0))**af
    Lp = 10.0 / af * np.log10(Af) - Lu + 94.0
    return f, Lp


def iso226_spl_contour_extended(Ln=40):
    if Ln < 0 or Ln > 90:
        raise AssertionError("Ln out of bounds [0, 90]")
    e_Af = 4.47e-3 * (10.0**(0.025 * Ln) - 1.15) + (0.4 * 10.0**((e_Tf + e_Lu) / 10.0 - 9.0))**e_af
    e_Lp = 10.0 / e_af * np.log10(e_Af) - e_Lu + 94.0
    return e_f, e_Lp


def iso226_loudness_contour(Lp=40):
    Bf = (0.4 * 10.0**((Lp + Lu) / 10.0 - 9.0))**af - (0.4 * 10.0**((Tf + Lu) / 10.0 - 9.0))**af + 0.005135
    Ln = 40.0 * np.log10(Bf) + 94.0
    return f, Ln


def iso226_loudness_contour_extended(Lp=40):
    e_Bf = (0.4 * 10.0**((Lp + e_Lu) / 10.0 - 9.0))**e_af - (0.4 * 10.0**((e_Tf + e_Lu) / 10.0 - 9.0))**e_af + 0.005135
    e_Ln = 40.0 * np.log10(e_Bf) + 94.0
    return e_f, e_Ln


def iso226_weights_db(frequencies, Lp=40):
    return interp1d(*iso226_loudness_contour_extended(Lp=Lp), "cubic")(frequencies)


def weigh(data, weights, axis):
    weights = np.power(10, (weights / 20))
    weights = weights / np.max(weights)
    return np.apply_along_axis(np.multiply, axis, data, weights)


def a_weigh_power_spectro(power_spectro, frequencies):
    return weigh(power_spectro, a_weights_db(frequencies), 1)


def iso226_weigh_power_spectro(power_spectro, frequencies):
    return weigh(power_spectro, iso226_weights_db(frequencies), 1)


def a_weigh_signal_coherence(signal_coherence, frequencies):
    return weigh(signal_coherence, a_weights_db(frequencies), 0)


def iso226_weigh_signal_coherence(signal_coherence, frequencies):
    return weigh(signal_coherence, iso226_weights_db(frequencies), 0)


if False:
    frequencies = np.geomspace(12, 24000, 2000)
    plt.plot(
        frequencies, iso226_weights_db(frequencies),
    )
    plt.xscale("symlog")
    plt.xlabel("frequencies")
    plt.ylabel("weights db")
    plt.grid(True, which="both")
    plt.xticks(ticks=[i * 10**e for e in range(5) for i in range(1, 10)])
    plt.legend()
    plt.show()


def difference_db(clear_signal, noisy_signal, q=None, is_power=False):
    v = np.abs(noisy_signal - clear_signal)
    if q is None:
        r = np.mean(v).item()
    else:
        r = np.percentile(v, q=q).item()
    return (10 if is_power else 20) * math.log10(r)


def noise_to_signal_ratio_db(clear_signal, noisy_signal, q=None, is_power=False):
    i = np.nonzero(clear_signal)  # avoid zero division loosing some data
    v = np.abs((noisy_signal[i] - clear_signal[i]) / clear_signal[i])
    if q is None:
        r = np.mean(v).item()
    else:
        r = np.percentile(v, q=q).item()
    return (10 if is_power else 20) * math.log10(r)


def calculate_difference(original_info, original, original_samplerate, decoded_info, decoded, decoded_samplerate, duration, original_spectro=None, weighted_original_spectro=None, differences=None):
    if differences is None:
        differences = {}
    if original_samplerate != decoded_samplerate:
        raise AssertionError("both original_samplerate and decoded_samplerate must be equal")
    if original_spectro is None != weighted_original_spectro is None:
        raise AssertionError("both original_spectro and weighted_original_spectro must be None or provided together")
    if "compression_ratio" not in differences:
        logging.info(">> Calculating decoded.wav compression_ratio ...")
        differences["compression_ratio"] = int(decoded_info["format"]["size"]) / int(original_info["format"]["size"])
    decoded_spectro = None
    weighted_decoded_spectro = None
    signal_coherence = None
    weighted_signal_coherence = None
    if "hash_perfect" not in differences or False:
        if "hash" not in original_info:
            logging.info(">> Hashing original.wav ...")
            original_info["hash"] = array_hash(original)
        if "hash" not in decoded_info:
            logging.info(">> Hashing decoded.wav ...")
            decoded_info["hash"] = array_hash(decoded)
        logging.info(">> Calculating decoded.wav hash_perfect ...")
        differences["hash_perfect"] = original_info["hash"] == decoded_info["hash"]
    # each variable has to check if has all it needs, it's ugly i know...
    if "signal_error_mean_db" not in differences or False:
        logging.info(">> Calculating decoded.wav signal_error_mean_db ...")
        differences["signal_error_mean_db"] = difference_db(original, decoded, is_power=False)
    if "signal_noise_to_signal_ratio_mean_db" not in differences or False:
        logging.info(">> Calculating decoded.wav signal_noise_to_signal_ratio_mean_db ...")
        differences["signal_noise_to_signal_ratio_mean_db"] = noise_to_signal_ratio_db(original, decoded, is_power=False)
    if "spectral_error_mean_db" not in differences or False:
        if original_spectro is None:
            logging.info(">> Calculating original.wav spectrogram ...")
            original_spectro_f, original_spectro_t, original_spectro = power_spectro(original, original_samplerate)
        if decoded_spectro is None:
            logging.info(">> Calculating decoded.wav spectrogram ...")
            decoded_spectro_f, decoded_spectro_t, decoded_spectro = power_spectro(decoded, decoded_samplerate)
            # decoded_spectro = decoded_spectro[:, :, :original_spectro.shape[2]]
        logging.info(">> Calculating decoded.wav spectral_error_mean_db ...")
        differences["spectral_error_mean_db"] = difference_db(original_spectro, decoded_spectro, is_power=True)
    if "weighted_spectral_error_mean_db" not in differences or False:
        if original_spectro is None:
            logging.info(">> Calculating original.wav spectrogram ...")
            original_spectro_f, original_spectro_t, original_spectro = power_spectro(original, original_samplerate)
        if decoded_spectro is None:
            logging.info(">> Calculating decoded.wav spectrogram ...")
            decoded_spectro_f, decoded_spectro_t, decoded_spectro = power_spectro(decoded, decoded_samplerate)
            # decoded_spectro = decoded_spectro[:, :, :original_spectro.shape[2]]
        if weighted_original_spectro is None:
            logging.info(">> Calculating original.wav weighted spectrogram ...")
            weighted_original_spectro = iso226_weigh_power_spectro(original_spectro, original_spectro_f)
        if weighted_decoded_spectro is None:
            logging.info(">> Calculating decoded.wav weighted spectrogram ...")
            weighted_decoded_spectro = iso226_weigh_power_spectro(decoded_spectro, decoded_spectro_f)
        logging.info(">> Calculating decoded.wav weighted_spectral_error_mean_db ...")
        differences["weighted_spectral_error_mean_db"] = difference_db(weighted_original_spectro, weighted_decoded_spectro, is_power=True)
    if "spectral_noise_to_signal_ratio_mean_db" not in differences or False:
        if original_spectro is None:
            logging.info(">> Calculating original.wav spectrogram ...")
            original_spectro_f, original_spectro_t, original_spectro = power_spectro(original, original_samplerate)
        if decoded_spectro is None:
            logging.info(">> Calculating decoded.wav spectrogram ...")
            decoded_spectro_f, decoded_spectro_t, decoded_spectro = power_spectro(decoded, decoded_samplerate)
            # decoded_spectro = decoded_spectro[:, :, :original_spectro.shape[2]]
        logging.info(">> Calculating decoded.wav spectral_noise_to_signal_ratio_mean_db ...")
        differences["spectral_noise_to_signal_ratio_mean_db"] = noise_to_signal_ratio_db(original_spectro, decoded_spectro, is_power=True)
    if "weighted_spectral_noise_to_signal_ratio_mean_db" not in differences or False:
        if original_spectro is None:
            logging.info(">> Calculating original.wav spectrogram ...")
            original_spectro_f, original_spectro_t, original_spectro = power_spectro(original, original_samplerate)
        if decoded_spectro is None:
            logging.info(">> Calculating decoded.wav spectrogram ...")
            decoded_spectro_f, decoded_spectro_t, decoded_spectro = power_spectro(decoded, decoded_samplerate)
            # decoded_spectro = decoded_spectro[:, :, :original_spectro.shape[2]]
        if weighted_original_spectro is None:
            logging.info(">> Calculating original.wav weighted spectrogram ...")
            weighted_original_spectro = iso226_weigh_power_spectro(original_spectro, original_spectro_f)
        if weighted_decoded_spectro is None:
            logging.info(">> Calculating decoded.wav weighted spectrogram ...")
            weighted_decoded_spectro = iso226_weigh_power_spectro(decoded_spectro, decoded_spectro_f)
        logging.info(">> Calculating decoded.wav weighted_spectral_noise_to_signal_ratio_mean_db ...")
        differences["weighted_spectral_noise_to_signal_ratio_mean_db"] = noise_to_signal_ratio_db(weighted_original_spectro, weighted_decoded_spectro, is_power=True)
    if "signal_incoherence" not in differences or False:
        if signal_coherence is None:
            logging.info(">> Calculating original.wav signal_coherence ...")
            signal_coherence_f, signal_coherence = signal_coherence_e(original, decoded, original_samplerate)
        logging.info(">> Calculating decoded.wav signal_incoherence ...")
        differences["signal_incoherence"] = signal_incoherence(signal_coherence)
    if "weighted_signal_incoherence" not in differences or False:
        if signal_coherence is None:
            logging.info(">> Calculating original.wav signal_coherence ...")
            signal_coherence_f, signal_coherence = signal_coherence_e(original, decoded, original_samplerate)
        if weighted_signal_coherence is None:
            logging.info(">> Calculating original.wav weighted_signal_coherence ...")
            weighted_signal_coherence = iso226_weigh_signal_coherence(signal_coherence, signal_coherence_f)
        logging.info(">> Calculating decoded.wav weighted_signal_incoherence ...")
        differences["weighted_signal_incoherence"] = signal_incoherence(weighted_signal_coherence)
    return differences, original_spectro, weighted_original_spectro


def merge_results(results, sep=" - "):
    """collisions can be possible and are not handled, avoid using sep in codec and spec names"""
    merged_results = {}
    for file, file_results in results.items():
        if file_results is None:
            continue
        for codec_name, codec_results in file_results.items():
            if codec_results is None:
                continue
            if codec_name not in merged_results:
                merged_results[codec_name] = {}
            for spec_name, spec_results in codec_results.items():
                if spec_results is None:
                    continue
                if spec_name not in merged_results[codec_name]:
                    merged_results[codec_name][spec_name] = {}
                for result_name, result_value in spec_results.items():
                    if result_name == "hash_perfect":
                        if result_name not in merged_results[codec_name][spec_name]:
                            merged_results[codec_name][spec_name][result_name] = result_value
                        else:
                            merged_results[codec_name][spec_name][result_name] &= result_value
                        continue
                    if result_name not in merged_results[codec_name][spec_name]:
                        merged_results[codec_name][spec_name][result_name] = []
                    merged_results[codec_name][spec_name][result_name].append(result_value)
    for codec_name, codec_results in merged_results.items():
        for spec_name, spec_results in codec_results.items():
            for result_name in spec_results:
                if result_name == "hash_perfect":
                    merged_results[codec_name][spec_name][result_name] = result_value
                    continue
                merged_results[codec_name][spec_name][result_name] = [mean(spec_results[result_name]), stdev(spec_results[result_name]) if len(spec_results[result_name]) > 1 else 0]
    return merged_results


def save(results, path_out):
    with open(path_out, "w") as f:
        json.dump(results, f, indent=" " * 4)


def delete(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def checkpoint(info, results):
    # print(info)
    # print(results)
    save(results, "results.json")
    save(info, "info.json")
    if False:
        with open("results.json", "r") as f:
            results = json.load(f)
    merged_results = merge_results(results)
    # print(merged_results)
    save(merged_results, "merged_results.json")
    return merged_results


def load_checkpoint():
    logging.error("> ----- Loading checkpoint -----")
    if os.path.exists("info.json"):
        with open("info.json", "r") as f:
            info = json.load(f)
    else:
        info = {}
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            results = json.load(f)
    else:
        results = {}
    if os.path.exists("merged_results.json"):
        with open("merged_results.json", "r") as f:
            merged_results = json.load(f)
    else:
        merged_results = {}
    return info, results, merged_results


exit_requested = False


def handler(signum, frame):
    global exit_requested
    exit_requested = True


def run_benchmark(files, codecs, resume=False):
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    # remove duplicate files and sort them
    files_set = set(files)
    files = sorted(files_set)
    total_to_run = sum(len(codec_specs) for codec_specs in codecs.values()) * len(files)
    logging.warning("---------- RUNNING BENCHMARKS ----------")
    if resume:
        info, results, merge_results = load_checkpoint()
    else:
        info, results, merge_results = {}, {}, {}
    # remove not selected files from info and results
    for file in set(list(info) + list(results)):
        if file not in files_set:
            info.pop(file, None)
            results.pop(file, None)
    to_run_n = 1
    for file_n, file in enumerate(files, start=1):
        progress = f"{file_n} of {len(files)} files, {to_run_n} of {total_to_run} to run"
        logging.warning(f"> ---------- Testing with {file!r} ({progress}) ----------")
        # decode source to common original.wav
        (
            original_samplerate,
            original,
            encoding_time,
            decoding_time,
        ) = encode_decode_load(
            None,
            samplerate=None,
            parameters=None,
            encoded_name=file,
            decoded_name="original.wav",
            samples=10,
            sample_lenght=1,
        )
        if False:
            save_audio_wav_pcm_s32le(original, original_samplerate, "tmp.wav")
        # check original is present and did not change or reset its info and results
        original_hash = array_hash(original)
        try:
            if original_hash != info[file]["original"]["original"]["hash"]:
                raise AssertionError("files differ")
        except (KeyError, AssertionError):
            info[file] = {
                "source": {"source": get_info(file)},
                "original": {"original": {**get_info("original.wav"), "hash": original_hash}}
            }
            results[file] = {}
        duration = float(info[file]["original"]["original"]["format"]["duration"])
        if False:
            logging.info(">> Calculating original.wav spectrogram ...")
            original_spectro_f, original_spectro_t, original_spectro = power_spectro(original, original_samplerate)
            logging.info(">> Plotting original.wav spectrogram ...")
            for channel in range(len(original_spectro)):
                plt.subplot()
                plt.pcolormesh(original_spectro_t, original_spectro_f, power_spectro_to_db(original_spectro[channel]), shading="gouraud")
                plt.yscale("symlog")
                plt.ylim(10, 30000)
                plt.ylabel("Frequency [Hz]")
                plt.xlabel("Time [sec]")
            plt.tight_layout()
            plt.show()
        else:
            original_spectro = None
        weighted_original_spectro = None
        if False:
            differences, original_spectro, weighted_original_spectro = calculate_difference(
                info[file]["original"]["original"],
                original,
                original_samplerate,
                info[file]["original"]["original"],
                original,
                original_samplerate,
                duration,
                original_power_spectro=original_spectro,
                weighted_original_spectro=weighted_original_spectro,
            )
            differences["encoding_time_per_second"] = encoding_time / duration
            differences["decoding_time_per_second"] = decoding_time / duration
            results[file]["original"] = {"original": differences}
        # checkpoint each loaded file
        merged_results = checkpoint(info, results)
        skipped = True
        for codec_name, codec_specs in codecs.items():
            for spec_name, parameters in codec_specs.items():
                global exit_requested
                if exit_requested:
                    quit()
                progress = f"{file_n} of {len(files)} files, {to_run_n} of {total_to_run} conversions to run"
                if codec_name not in info[file]:
                    info[file][codec_name] = {}
                if codec_name not in results[file]:
                    results[file][codec_name] = {}
                # check parameters are present and did not change or reset its info and results
                try:
                    if parameters != info[file][codec_name][spec_name]["_parameters"]:
                        raise AssertionError("parameters differ")
                except (KeyError, AssertionError):
                    info[file][codec_name][spec_name] = {"_parameters": parameters}
                    results[file][codec_name][spec_name] = None  # mark as to be run
                if results[file][codec_name][spec_name] is not None:
                    if all(i in results[file][codec_name][spec_name] for i in {
                        "signal_error_mean_db",
                        "signal_noise_to_signal_ratio_mean_db",
                        "spectral_error_mean_db",
                        "weighted_spectral_error_mean_db",
                        "spectral_noise_to_signal_ratio_mean_db",
                        "weighted_spectral_noise_to_signal_ratio_mean_db",
                        "signal_incoherence",
                        "weighted_signal_incoherence",
                    }) and True:
                        # logging.warning(f">> ----- Skipping {codec_name!r} {spec_name!r} with parameters: {parameters!r} ({progress}) -----")
                        logging.warning(f">> ----- Skipping {codec_name!r} {spec_name!r} ({progress}) -----")
                        to_run_n += 1
                        continue  # do not run again if results are present
                # logging.warning(f">> ----- Running {codec_name!r} {spec_name!r} with parameters: {parameters!r} ({progress}) -----")
                logging.warning(f">> ----- Running {codec_name!r} {spec_name!r} ({progress}) -----")
                try:
                    (
                        decoded_samplerate,
                        decoded,
                        encoding_time,
                        decoding_time,
                    ) = encode_decode_load(
                        "original.wav",
                        samplerate=original_samplerate,
                        parameters=parameters,
                    )
                    if decoded.shape[0] < original.shape[0]:
                        decoded = np.vstack((decoded, np.zeros((original.shape[0] - decoded.shape[0], *original.shape[1:]))))
                    elif decoded.shape[0] > original.shape[0]:
                        decoded = decoded[:original.shape[0]]
                    info[file][codec_name][spec_name].update(get_info("encoded"))
                    logging.info(">> Calculating decoded.wav difference from original.wav ...")
                    differences, original_spectro, weighted_original_spectro = calculate_difference(
                        info[file]["original"]["original"],
                        original,
                        original_samplerate,
                        info[file][codec_name][spec_name],
                        decoded,
                        decoded_samplerate,
                        duration,
                        original_spectro=original_spectro,
                        weighted_original_spectro=weighted_original_spectro,
                        differences=results[file][codec_name][spec_name],
                    )
                    differences["encoding_time_per_second"] = encoding_time / duration
                    differences["decoding_time_per_second"] = decoding_time / duration
                    results[file][codec_name][spec_name] = differences
                    do_checkpoint = True
                    skipped = False
                except Exception:
                    logging.exception(traceback.format_exc())
                    do_checkpoint = False
                if do_checkpoint:
                    # checkpoint each completed result
                    merged_results = checkpoint(info, results)
                else:
                    # or reload previous results on fail
                    info, results, merge_results = load_checkpoint()
                to_run_n += 1
        if not skipped:
            save_table(merged_results)
            plot_merged_results(merged_results)
    delete("original.wav")
    delete("encoded")
    delete("decoded.wav")
    signal.signal(signal.SIGINT, original_sigint_handler)
    return merged_results


def save_table(merged_results):
    logging.warning("---------- SAVING RESULTS TABLE ----------")
    version_string = ffmpeg.input("").output("").run(cmd=["ffmpeg", "-version"], capture_stdout=True)[0]
    version_string = version_string.decode("utf8").split("\n")[0]
    version = re.search(r"^ffmpeg version ([^ ]*) Copyright \(c\) .*", version_string).group(1)
    with open("merged_results.md", "w") as f:
        f.write(f"""
# Criteria\n
The tests have been run using, [`ffmpeg`](https://ffmpeg.org/) ({version} on {platform.system()}-{platform.machine()}).
- *Format*: The format used to generate encoded from the original
- *Spec*: Name of the set of parameters used to generate encoded from the original
- *Compression Ratio (%)*: `encoded_matroska_file_size / original_wav_file_size`
- *Hash perfect*: `encoded_array_hash == original_array_hash`
- *Encoding time (s)*: time to convert one second of original (wav) to encoded (matroska)
- *Decoding time (s)*: time to convert one second of encoded (matroska) to decoded (wav)
- *Signal Error (db)*: `20 * log10(mean(abs(endoded - original)))` (lower is better)
- *Signal Noise-to-Signal Ratio (db)*: `10 * log10(mean((decoded - original) / original))` (lower is better)
- *Spectral Error (db)*: `20 * log10(mean(abs(power_spectro(decoded, decoded_samplerate) - power_spectro(original, original_samplerate))))` (lower is better)
- *Weighted Spectral Error (db)*: Same as Spectral Error but the spectrogram is weighted with iso226_weigh_power_spectro (lower is better)
- *Spectral Noise-to-Signal Ratio (db)*: `10 * log10(mean((power_spectro(decoded, decoded_samplerate) - power_spectro(original, original_samplerate)) / power_spectro(original, original_samplerate)))` (lower is better)
- *Weighted Spectral Noise-to-Signal Ratio (db)*: Same as Spectral Noise-to-Signal Ratio but the spectrogram is weighted with iso226_weigh_power_spectro (lower is better)
- *Signal Incoherence*: `1 - mean(coherence(original, decoded, original_samplerate))` (lower is better)
- *Weighted Signal Incoherence*: Same as Signal Incoherence but the coherence is weighted with iso226_weigh_signal_coherence (lower is better)
""")

        f.write("# Results\n\n")
        f.write("Results have format `mean (stdev)`\n")
        f.write('''
<table>
<tr>
\t<th rowspan="2">Format</th>
\t<th rowspan="2">Spec</th>
\t<th rowspan="2">Compression Ratio (%)</th>
\t<th rowspan="2">Hash perfect</th>
\t<th rowspan="2">Encoding time (s)</th>
\t<th rowspan="2">Decoding time (s)</th>
\t<th rowspan="2">Signal Error (db)</th>
\t<th rowspan="2">Signal Noise-to-Signal Ratio (db)</th>
\t<th colspan="2">Spectral Error (db)</th>
\t<th colspan="2">Spectral Noise-to-Signal Ratio (db)</th>
\t<th colspan="2">Signal Incoherence</th>
</tr>
<tr>
\t<th>linear</th>
\t<th>weighted</th>
\t<th>linear</th>
\t<th>weighted</th>
\t<th>linear</th>
\t<th>weighted</th>
</tr>
''')
        for codec_name, codec_results in merged_results.items():
            first = True
            for spec_name, spec_results in codec_results.items():
                f.write('<tr>\n')
                if first:
                    f.write(f'<td rowspan="{len(codec_results)}">{codec_name}</td>\n')
                    first = False
                f.write(f'\t<td>{spec_name}</td>\n')
                f.write(f'\t<td>{spec_results["compression_ratio"][0] * 100:.3f}% ({spec_results["compression_ratio"][1] * 100:.3f})</td>\n')
                f.write(f'\t<td>{"yes" if spec_results["hash_perfect"] is True else "no"}</td>\n')
                for n in [
                    "encoding_time_per_second",
                    "decoding_time_per_second",
                    "signal_error_mean_db",
                    "signal_noise_to_signal_ratio_mean_db",
                    "spectral_error_mean_db",
                    "weighted_spectral_error_mean_db",
                    "spectral_noise_to_signal_ratio_mean_db",
                    "weighted_spectral_noise_to_signal_ratio_mean_db",
                    "signal_incoherence",
                    "weighted_signal_incoherence",
                ]:
                    if n not in spec_results:
                        continue
                    f.write(f'\t<td>{spec_results[n][0]:.3f} ({spec_results[n][1]:.3f})</td>\n')
            f.write('</tr>\n')
        f.write('</table>\n')
        f.write('''
# Plotted results\n
![Quality](./figures/quality.png)
''')


def plot_merged_results(merged_results):
    logging.warning("---------- SAVING RESULTS PLOTS ----------")
    fig = Path(".") / "figures"
    plt.figure(figsize=[mm / 25.4 for mm in [210, 297]])
    for codec_name, codec_results in merged_results.items():
        codec_plots = {}
        for spec_name, spec_results in codec_results.items():
            for result_name, result_value in spec_results.items():
                if result_name == "hash_perfect":
                    continue
                if result_name not in codec_plots:
                    codec_plots[result_name] = [[], []]
                codec_plots[result_name][0].append(result_value[0])
                codec_plots[result_name][1].append(result_value[1])
        compression_ratio = [[i * 100 for i in ms] for ms in codec_plots["compression_ratio"]]
        for i, n in enumerate([
            "signal_error_mean_db",
            "signal_noise_to_signal_ratio_mean_db",
            "spectral_error_mean_db",
            "spectral_noise_to_signal_ratio_mean_db",
            "weighted_spectral_error_mean_db",
            "weighted_spectral_noise_to_signal_ratio_mean_db",
            "signal_incoherence",
            "weighted_signal_incoherence",
        ], start=1):
            if n not in codec_plots:
                continue
            plt.subplot(4, 2, i)
            plt.errorbar(
                x=compression_ratio[0],
                xerr=compression_ratio[1],
                y=codec_plots[n][0],
                yerr=codec_plots[n][1],
                marker="o",
                markersize=2.50,
                linewidth=1.00,
                elinewidth=0.25,
                label=codec_name,
            )
            # set multiple times to make the code shorter
            plt.xlabel("compression ratio %")
            plt.ylabel(n.replace("_", " "))
            plt.legend()
            plt.tight_layout()
    plt.savefig(fig / "quality.png", dpi=300)
    if False:
        plt.show()
