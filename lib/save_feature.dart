import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:google_speech/google_speech.dart';
import 'package:fftea/fftea.dart';
import 'package:xml/xml.dart' as xml;
import 'svcNoFreq.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Save Features',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SoundRecorder(),
    );
  }
}

class SoundRecorder extends StatefulWidget {
  @override
  _SoundRecorderState createState() => _SoundRecorderState();
}

class _SoundRecorderState extends State<SoundRecorder> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final speechToText = SpeechToText.viaApiKey("AIzaSyDx5fZpE0z1QxYV7mwN1cvxig7tUvzw4Xc");
  final config = RecognitionConfig(
    encoding: AudioEncoding.LINEAR16,
    model: RecognitionModel.basic,
    enableAutomaticPunctuation: true,
    sampleRateHertz: 16000,
    languageCode: 'id-ID',
  );
  final List<String> keywords = ['tolong', 'help', 'tolong', 'aw', 'help','aduh', 'aw', 'aduh'];
  bool showError = false;
  bool _isRecording = false;
  int recordingCount = 0;
  // int currentRecordingIndex = 0;
  String? _filePath;
  List<Map<String, dynamic>> trainingData = [];
  List<double> svmWeights = [];
  double svmBias = 0.0;
  List<double> bestWeights = [];
  double bestBias = 0.0;
  double bestLearningRate = 0.0;
  int bestEpoch = 0;
  double bestC = 0.0;

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    // Request permissions for microphone and storage
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      print('Microphone permission not granted');
      return;
    }
    await _recorder.openRecorder();
    _recorder.setSubscriptionDuration(Duration(milliseconds: 2000));
  }
  // =================== MFCC ======================
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<String> _extractWaveform(String inputPath) async {
    String outputPath = '${inputPath}_waveform.pcm';
    String command = '-y -i "$inputPath" -ar 16000 -ac 1 -f s16le "$outputPath"';

    await FFmpegKit.execute(command).then((session) async {
      final returnCode = await session.getReturnCode();
      if (returnCode != null && returnCode.isValueSuccess()) {
        print('Waveform extracted successfully for $inputPath');
      } else {
        final output = await session.getOutput();
        print('Error extracting waveform: $output');
      }
    });

    return outputPath;
  }

  Future<List<int>> _getAudioBytes(String filePath) async {
    final audioFile = File(filePath);
    if (!await audioFile.exists()) {
      throw Exception("Audio file not found at path: $filePath");
    }
    final audioData = await audioFile.readAsBytes();
    return audioData;
  }

  List<double> normalizeAudioData(List<int> audioBytes) {
    List<double> normalizedData = [];
    for (int i = 0; i < audioBytes.length - 1; i += 2) {
      int sample = audioBytes[i] | (audioBytes[i + 1] << 8);
      if (sample > 32767) sample -= 65536;
      normalizedData.add(sample / 32768.0);
    }

    // if (normalizedData.every((sample) => sample == 0)) {
    //   throw Exception("Audio normalization failed. All samples are zero.");
    // }

    return normalizedData;
  }

  List<Float64List> melFilterbank(int numFilters, int fftSize, int sampleRate) {
    // Helper function to convert frequency to Mel scale
    double hzToMel(double hz) {
      return 2595 * log(1 + hz / 700) / ln10; // Convert Hz to Mel scale
    }

    // Helper function to convert Mel scale to frequency
    double melToHz(double mel) {
      return 700 * (pow(10, mel / 2595) - 1); // Convert Mel scale to Hz
    }

    // Create filterbank
    var melFilters = List<Float64List>.generate(numFilters, (i) => Float64List(fftSize ~/ 2 + 1));

    // Define the low and high frequency limits
    double lowFreqMel = hzToMel(0); // Lowest frequency (0 Hz)
    double highFreqMel = hzToMel(sampleRate / 2); // Nyquist frequency (half of sample rate)

    // Compute equally spaced Mel points
    var melPoints = List<double>.generate(numFilters + 2, (i) {
      return lowFreqMel + i * (highFreqMel - lowFreqMel) / (numFilters + 1);
    });

    // Convert Mel points back to Hz
    var hzPoints = melPoints.map(melToHz).toList();

    // Convert Hz points to FFT bin numbers
    var binPoints = hzPoints.map((hz) {
      return ((fftSize + 1) * hz / sampleRate).floor();
    }).toList();

    // Fill the Mel filterbank with triangular filters
    for (int i = 1; i < numFilters + 1; i++) {
      for (int j = binPoints[i - 1]; j < binPoints[i]; j++) {
        melFilters[i - 1][j] = (j - binPoints[i - 1]) / (binPoints[i] - binPoints[i - 1]);
      }
      for (int j = binPoints[i]; j < binPoints[i + 1]; j++) {
        melFilters[i - 1][j] = (binPoints[i + 1] - j) / (binPoints[i + 1] - binPoints[i]);
      }
    }

    return melFilters;
  }

  List<double> applyMelFilterbank(List<double> stftFrame, List<Float64List> melFilters) {
    var melEnergies = List<double>.filled(melFilters.length, 0.0);

    for (int i = 0; i < melFilters.length; i++) {
      melEnergies[i] = dot(melFilters[i], stftFrame);
    }

    return melEnergies;
  }

  double dot(List<double> vectorA, List<double> vectorB) {
    if (vectorA.length != vectorB.length) {
      throw Exception('Vector lengths must be equal for dot product');
    }

    double result = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      result += vectorA[i] * vectorB[i];
    }
    return result;
  }

  List<double> dct(List<double> input, int numCoefficients) {
    int N = input.length;
    List<double> output = List<double>.filled(numCoefficients, 0.0);

    for (int k = 0; k < numCoefficients; k++) {
      double sum = 0.0;
      for (int n = 0; n < N; n++) {
        sum += input[n] * cos((pi / N) * (n + 0.5) * k);
      }
      output[k] = sum;
    }

    return output;
  }

  List<double> computeMFCC(List<int> audioBytes, int sampleRate, int numCoefficients) {
    var audioSignal = normalizeAudioData(audioBytes);

    final chunkSize = 512;
    final stft = STFT(chunkSize, Window.hanning(chunkSize));
    final spectrogram = <Float64List>[];

    stft.run(audioSignal, (Float64x2List freq) {
      final magnitudes = freq.discardConjugates().magnitudes();
      spectrogram.add(magnitudes);
    });

    var melFilters = melFilterbank(26, chunkSize, sampleRate);
    var melSpectrogram = <List<double>>[];
    for (var frame in spectrogram) {
      var melEnergies = applyMelFilterbank(frame, melFilters);
      melSpectrogram.add(melEnergies);
    }

    for (var i = 0; i < melSpectrogram.length; i++) {
      for (var j = 0; j < melSpectrogram[i].length; j++) {
        melSpectrogram[i][j] = log(melSpectrogram[i][j] + 1e-10);
      }
    }


    var mfccList = <double>[];
    for (var frame in melSpectrogram) {
      var dctCoeffs = dct(frame, numCoefficients);
      mfccList.addAll(dctCoeffs);
    }

    return mfccList;
  }
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // ================================================

  // ============== EKSTRAKSI FITUR =================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<Map<String, dynamic>> extractAudioFeatures(String filePath) async {
    // Read the file
    File file = File(filePath);
    Uint8List audioData = await file.readAsBytes();

    // Convert Uint8List to List<int> (assuming 16-bit PCM)
    List<int> audioListInt = [];
    for (int i = 0; i < audioData.length; i += 2) {
      int value = (audioData[i + 1] << 8) | audioData[i];
      if (value >= 0x8000) value -= 0x10000;
      audioListInt.add(value);
    }

    // Convert to List<double>
    List<double> audioList = audioListInt.map((e) => e.toDouble()).toList();

    // Remove DC component (mean removal)
    double mean = audioList.reduce((a, b) => a + b) / audioList.length;
    audioList = audioList.map((v) => v - mean).toList();

    // Normalize amplitude
    double maxAbsVal = audioList.map((v) => v.abs()).reduce((a, b) => a > b ? a : b);
    if (maxAbsVal > 0) {
      audioList = audioList.map((v) => v / maxAbsVal).toList();
    }

    // FFT to calculate dominant frequency
    final fft = FFT(audioList.length);
    final freqs = fft.realFft(audioList);

    // Spectral magnitude
    List<double> freqsDouble = freqs.map((f) => sqrt(f.x * f.x + f.y * f.y)).toList();
    double maxAmplitude = freqsDouble.reduce((curr, next) => curr.abs() > next.abs() ? curr : next);
    int maxFreqIndex = freqsDouble.indexOf(maxAmplitude);

    // Dominant frequency calculation
    double dominantFreq = (maxFreqIndex * 16000) / (audioList.length / 2); // Divide by 2 for FFT symmetry

    // Decibel calculation
    double minAmplitude = 1e-10;
    maxAmplitude = maxAmplitude.abs();
    if (maxAmplitude < minAmplitude) {
      maxAmplitude = minAmplitude;
    }
    double decibel = 20 * log(maxAmplitude) / log(10);

    // MFCC Calculation (13 coefficients as an example)
    String pcmPath = await _extractWaveform(filePath);
    final audioBytes = await _getAudioBytes(pcmPath);
    List<double> sampleMFCC = computeMFCC(audioBytes, 16000, 13);

    // Return the extracted features as a map
    return {
      'frequency': dominantFreq,
      'amplitude': maxAmplitude,
      'decibel': decibel,
      'mfcc': sampleMFCC,
    };
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ==================LOAD XML =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> loadFeaturesFromXml() async {
    // Tentukan lokasi file XML yang disimpan di penyimpanan eksternal
    final directory = await getExternalStorageDirectory();
    final filePath = '${directory?.path}/audio_features.xml';

    // Baca konten file XML dari lokasi penyimpanan eksternal
    final file = File(filePath);
    if (!file.existsSync()) {
      print('File tidak ditemukan: $filePath');
      return;
    }

    final xmlString = await file.readAsString();
    final xmlDoc = xml.XmlDocument.parse(xmlString);

    // Extract data from the XML
    for (var audioElement in xmlDoc.findAllElements('Audio')) {
      // Ubah label dari string menjadi integer
      int label = int.parse(audioElement.findElements('Label').first.text);

      double frequency = double.parse(audioElement.findElements('Frequency').first.text);
      double amplitude = double.parse(audioElement.findElements('Amplitude').first.text);
      double decibel = double.parse(audioElement.findElements('Decibel').first.text);
      List<double> mfcc = audioElement.findElements('MFCC')
          .map((e) => double.parse(e.text))
          .toList();

      trainingData.add({
        'label': label,  // Sekarang label disimpan sebagai integer
        'features': {
          'frequency': frequency,
          'amplitude': amplitude,
          'decibel': decibel,
          'mfcc': mfcc,
        }
      });
    }

    print('Audio features loaded successfully.');
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ============= Make XML ===================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> saveFeaturesToXml(List<Map<String, dynamic>> audioStore, String filePath) async {
    xml.XmlDocument xmlDocument;

    // Cek apakah file sudah ada
    final file = File(filePath);
    if (file.existsSync()) {
      final xmlString = await file.readAsString();
      xmlDocument = xml.XmlDocument.parse(xmlString);
    } else {
      // Membuat dokumen XML baru jika file belum ada
      xmlDocument = xml.XmlDocument([
        xml.XmlProcessing('xml', 'version="1.0"'),
        xml.XmlElement(xml.XmlName('AudioFeatures')),
      ]);
    }

    // Menambahkan fitur audio baru ke elemen root 'AudioFeatures'
    final rootElement = xmlDocument.rootElement;
    for (var audioData in audioStore) {
      final audioElement = xml.XmlElement(xml.XmlName('Audio'));

      audioElement.children.add(xml.XmlElement(xml.XmlName('Label'), [], [xml.XmlText(audioData['label'].toString())]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Frequency'), [], [xml.XmlText(audioData['features']['frequency'].toString())]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Amplitude'), [], [xml.XmlText(audioData['features']['amplitude'].toString())]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Decibel'), [], [xml.XmlText(audioData['features']['decibel'].toString())]));
      List mfccValues = audioData['features']['mfcc'].toList();
      for (var mfcc in mfccValues) {
        audioElement.children.add(xml.XmlElement(xml.XmlName('MFCC'), [], [xml.XmlText(mfcc.toString())]));
      }

      rootElement.children.add(audioElement);
    }

    // Menyimpan dokumen XML yang sudah diperbarui ke file yang sama
    await file.writeAsString(xmlDocument.toXmlString(pretty: true));
    print('New audio features have been appended to the file at: $filePath');
  }

  Future<void> loadAndAppendFeaturesFromAssets(List<Map<String, dynamic>> audioStore, String filePath) async {
    for (int i = 1; i <= 5; i++) {
      // Muat file audio dari assets
      ByteData audioData = await rootBundle.load('assets/audio/test$i.wav');

      // Simpan sementara sebagai file untuk proses ekstraksi
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir?.path}/test$i.wav');
      await tempFile.writeAsBytes(audioData.buffer.asUint8List());

      // Ekstrak fitur audio
      Map<String, dynamic> features = await extractAudioFeatures(tempFile.path);

      // Tambahkan ke audioStore dengan label yang sesuai
      audioStore.add({
        'label': -1,
        'features': features,
      });
    }

    // Setelah semua fitur dari assets ditambahkan, simpan ke file XML
    await saveFeaturesToXml(audioStore, filePath);
    print('All audio features from assets have been saved to XML file.');
  }

  Future<void> saveSVMParametersToXml(List<double> weights, double bias) async {
    final builder = xml.XmlBuilder();
    builder.processing('xml', 'version="1.0"');
    builder.element('SVMParameters', nest: () {
      builder.element('Weights', nest: () {
        for (var weight in weights) {
          builder.element('Weight', nest: weight.toString());
        }
      });
      builder.element('Bias', nest: bias.toString());
    });

    final directory = await getExternalStorageDirectory();
    final filePath = '${directory!.path}/svm_parameters.xml';
    final file = File(filePath);
    file.writeAsStringSync(builder.buildDocument().toString());

    print('Saved best SVM parameters to: $filePath');
  }

  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =================== SVM ========================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // Fungsi untuk melatih SVM dengan hyperparameter yang diberikan
  void _trainSVM(List<Map<String, dynamic>> trainingData, double learningRate, int numEpochs, double C) {
    int fixedMfccLength = 200;
    int numFeatures = fixedMfccLength + 2;
    svmWeights = List<double>.filled(numFeatures, 0.0);
    svmBias = 0.0;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
      for (var sample in trainingData) {
        List<double> features = [
          sample['features']['amplitude'],
          sample['features']['decibel']
        ];

        List<double> mfccs = sample['features']['mfcc'];
        if (mfccs.length > fixedMfccLength) {
          mfccs = mfccs.sublist(0, fixedMfccLength);
        } else if (mfccs.length < fixedMfccLength) {
          mfccs = List<double>.from(mfccs)..addAll(List<double>.filled(fixedMfccLength - mfccs.length, 0.0));
        }
        features.addAll(mfccs);
        int label = sample['label'];
        double prediction = _svmPredict(features);
        double margin = label * prediction;

        if (margin < 1) {
          double slack = 1 - margin;
          for (int i = 0; i < svmWeights.length; i++) {
            svmWeights[i] += learningRate * (label * features[i] - 2 * 0.01 * svmWeights[i]);
          }
          svmBias += learningRate * (label - C * slack);
        } else {
          for (int i = 0; i < svmWeights.length; i++) {
            svmWeights[i] += learningRate * (-2 * 0.01 * svmWeights[i]);
          }
        }
      }
    }
  }

  double _svmPredict(List<double> features) {
    int minLength = min(svmWeights.length, features.length);
    List<double> trimmedFeatures = features.sublist(0, minLength);
    List<double> trimmedWeights = svmWeights.sublist(0, minLength);

    double result = svmBias;
    for (int i = 0; i < trimmedWeights.length; i++) {
      result += trimmedWeights[i] * trimmedFeatures[i];
    }
    return result;
  }

  double evaluateSVM(List<Map<String, dynamic>> validationData) {
    int correctPredictions = 0;
    for (var sample in validationData) {
      List<double> features = [
        sample['features']['amplitude'],
        sample['features']['decibel'],
        ...sample['features']['mfcc']
      ];
      int label = sample['label'];
      double prediction = _svmPredict(features);
      if ((prediction > 0 && label == 1) || (prediction <= 0 && label == -1)) {
        correctPredictions++;
      }
    }
    return correctPredictions / validationData.length;
  }

  void tuneAndTrainSVM(List<Map<String, dynamic>> trainingData, List<Map<String, dynamic>> validationData) {
    List<double> learningRates = [0.001, 0.01, 0.1];
    List<int> epochOptions = [50, 100, 150];
    List<double> penaltyValues = [0.1, 1.0, 10.0];

    double bestAccuracy = 0.0;

    for (double learningRate in learningRates) {
      for (int numEpochs in epochOptions) {
        for (double C in penaltyValues) {
          _trainSVM(trainingData, learningRate, numEpochs, C);
          double accuracy = evaluateSVM(validationData);

          if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestLearningRate = learningRate;
            bestEpoch = numEpochs;
            bestC = C;
            bestWeights = List.from(svmWeights);
            bestBias = svmBias;
          }
        }
      }
    }

    print('Best Parameters - Learning Rate: $bestLearningRate, Epochs: $bestEpoch, C: $bestC');
    print('Best Accuracy: $bestAccuracy');

    // Simpan parameter terbaik ke dalam file XML
    saveSVMParametersToXml(bestWeights, bestBias);
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =================== RECORD =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> _startRecording() async {
    try {
      if (recordingCount >= 8) {
        print('maximum recording reached.');
        return;
      }

      final directory = await getExternalStorageDirectory();
      _filePath = '${directory?.path}/audio_record.wav';
      print('Audio saved to: $_filePath');

      try {
        await _recorder.startRecorder(
          toFile: _filePath,
          codec: Codec.pcm16WAV,
        );
      } catch (e) {
        print('Error in starting recorder: $e');
      }

      var status = await Permission.microphone.request();
      if (!status.isGranted) {
        print("Microphone permission is not granted");
        return; // Jika izin tidak diberikan, tidak melanjutkan perekaman
      }

      setState(() {
        _isRecording = true;
      });

      Timer(Duration(seconds: 2), () async {
        await _stopRecording();
      });
    } catch (e) {
      print('Error starting recording: $e');
    }
  }

  Future<List<int>> _getAudioContent(String path) async {
    return File(path).readAsBytesSync().toList();
  }

  // Modifikasi pada fungsi _stopRecording
  Future<void> _stopRecording() async {
    try {
      if (_recorder.isRecording) {
        await _recorder.stopRecorder();
        print('Recording stopped.');
      }

      setState(() {
        _isRecording = false;
        showError = false;
      });

      final audio = await _getAudioContent(_filePath!);
      final response = await speechToText.recognize(config, audio);
      String? detectedText = response.results
          .map((result) => result.alternatives.first.transcript)
          .join(' ');

      if (detectedText == keywords[recordingCount]) {
        setState(() {
          recordingCount++;
          // currentRecordingIndex++;
        });
        print("Correct keyword detected: $detectedText");

        List<Map<String, dynamic>> audioStore = [];

        if (_filePath != null) {
          Map<String, dynamic> recordedFeatures = await extractAudioFeatures(_filePath!);
          audioStore.add({
            'label': 1,
            'features': recordedFeatures,
          });
        }

        // Simpan fitur dari rekaman ke file XML
        final directory = await getExternalStorageDirectory();
        if (directory != null) {
          String filePath = '${directory.path}/audio_features.xml';
          await saveFeaturesToXml(audioStore, filePath);
          print('Audio features saved successfully.');
        }

        // add asset if record count is 8
        if (recordingCount == 8) {
          final directory = await getExternalStorageDirectory();
          if (directory != null) {
            String filePath = '${directory.path}/audio_features.xml';
            await loadAndAppendFeaturesFromAssets([], filePath); // Menggunakan array kosong agar tidak duplikasi
            print('Audio features from assets saved successfully.');

            // Load features from XML and train SVM
            await loadFeaturesFromXml(); // Memuat data fitur dari file XML ke dalam trainingData
            tuneAndTrainSVM(trainingData, trainingData); // Melatih model SVM
            print('SVM training completed.');
          }
        }
      } else {
        setState(() {
          showError = true;
        });
        // Incorrect keyword, prompt to retry
        print("Incorrect keyword. Expected: ${keywords[recordingCount]}, Detected: $detectedText");
      }
    } catch (e) {
      print('Error stopping recorder: $e');
    }
  }

  Future<void> _resetRecording() async {
    try {
      // Get the directory and XML file path
      final directory = await getExternalStorageDirectory();
      if (directory != null) {
        final filePathFeature = '${directory.path}/audio_features.xml';
        final filefeature = File(filePathFeature);
        final filePathSvm = '${directory.path}/svm_parameters.xml';
        final filesvm = File(filePathSvm);

        // Delete the XML file if it exists
        if (await filefeature.exists()) {
          await filefeature.delete();
          print('XML Feature file deleted.');
        }
        if (await filesvm.exists()) {
          await filesvm.delete();
          print('XML SVM file deleted.');
        }
      }

      // Reset recording count and error state
      setState(() {
        recordingCount = 0;
        showError = false;
      });

      print('Recording count reset.');
    } catch (e) {
      print('Error resetting recording: $e');
    }
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Save Features'),
        centerTitle: true, // Center the title for a clean design
        backgroundColor: Colors.teal, // Custom app bar color
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center, // Align all widgets centrally
            children: <Widget>[
              // Display Keyword Prompt
              if (recordingCount < keywords.length)
                Text(
                  'Ucapkan Kata: "${keywords[recordingCount]}"',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold, // Emphasize this text
                    color: Colors.blueGrey[800],
                  ),
                )
              else
                Text(
                  'Record sudah sesuai',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.green,
                  ),
                ),
              SizedBox(height: 30), // Add more space between elements

              // Microphone Button with subtle animation
              GestureDetector(
                onTap: _isRecording || recordingCount >= 8 ? null : _startRecording,
                child: AnimatedContainer(
                  duration: Duration(milliseconds: 300),
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: _isRecording ? Colors.redAccent : Colors.brown[400],
                    shape: BoxShape.circle,
                    boxShadow: _isRecording
                        ? [
                      BoxShadow(
                        color: Colors.redAccent.withOpacity(0.5),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ]
                        : [],
                  ),
                  child: Icon(
                    _isRecording ? Icons.mic : Icons.mic_off,
                    color: Colors.white,
                    size: 40,
                  ),
                ),
              ),
              SizedBox(height: 20), // Adjust spacing between elements

              // Error message
              if (showError)
                Text(
                  'Kata tidak sesuai, coba lagi.', // Incorrect keyword message
                  style: TextStyle(
                    color: Colors.red,
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                )
              else if (showError = true) // Show only if there is successful detection
                Text(
                  'Voice Correct', // Correct keyword message
                  style: TextStyle(
                    color: Colors.green,  // Green font color
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              // Text(
              //     'Rekam lagi !!!',
              //   style: TextStyle(
              //     color: Colors.blueGrey,
              //     fontSize: 12,
              //   ),),
              SizedBox(height: 20),

              // Recording progress text
              Text(
                'Recording ${recordingCount}/8',
                style: TextStyle(fontSize: 18, color: Colors.teal),
              ),
              SizedBox(height: 20),

              // Progress bar to visually show recording count
              LinearProgressIndicator(
                value: recordingCount / 8, // Progress bar update
                backgroundColor: Colors.grey[300],
                color: Colors.teal,
                minHeight: 10,
              ),
              SizedBox(height: 30),

              // Reset Button
              ElevatedButton(
                onPressed: _resetRecording,
                child: Text("Reset Recording"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(horizontal: 30, vertical: 10),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}