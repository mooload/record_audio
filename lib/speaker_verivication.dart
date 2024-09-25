import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:fftea/fftea.dart';
import 'package:scidart/numdart.dart';
import 'package:audio_recording/main.dart';

class SpeakerVerification {
  Interpreter? _interpreter;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('erlan_matching.tflite');
      print('Model berhasil dimuat');
    } catch (e) {
      print('Error saat memuat model: $e');
    }
  }

  Future<List<double>> extractMFCC(List<double> signal, int sampleRate, int numCoeffs) async {
    // 1. Terapkan FFT pada sinyal
    final fft = FFT(signal.length);
    final freqs = fft.realFft(signal);

    // 2. Konversi frekuensi ke skala Mel
    var melFilters = generateMelFilterBank(numCoeffs, freqs.length, sampleRate);

    // 3. Ambil log dari energi filter bank
    var logEnergies = melFilters.map((f) => log(f.abs() + 1e-10)).toList();

    // 4. Terapkan DCT ke log Mel energies untuk mendapatkan MFCCs
    var mfccs = dct(logEnergies);

    // Batasi hasil DCT ke 13 koefisien
    if (mfccs.length > 13) {
      mfccs = mfccs.sublist(0, 13);
    }

    return mfccs;
  }

  List<double> calculateFeatureDifference(List<double> mfcc1, List<double> mfcc2) {
    List<double> featureDifference = [];
    for (int i = 0; i < mfcc1.length; i++) {
      featureDifference.add((mfcc1[i] - mfcc2[i]).abs());
    }
    return featureDifference;
  }

  Future<double?> predictSpeakerVerification(String file1, String file2) async {
    if (_interpreter == null) {
      print("Model belum dimuat");
      return null;
    }

    List<double> mfcc1 = await extractMFCC(file1);
    List<double> mfcc2 = await extractMFCC(file2);
    List<double> featureDifference = calculateFeatureDifference(mfcc1, mfcc2);

    var input = featureDifference.reshape([1, featureDifference.length]);
    var output = List.filled(1, 0).reshape([1]);

    _interpreter?.run(input, output);

    return output[0];
  }
}
