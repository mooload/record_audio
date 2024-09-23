import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:fftea/fftea.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:math';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Recorder',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: AudioRecorderHome(),
    );
  }
}

class AudioRecorderHome extends StatefulWidget {
  @override
  _AudioRecorderHomeState createState() => _AudioRecorderHomeState();
}

class _AudioRecorderHomeState extends State<AudioRecorderHome> {
  FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool isRecording = false; // State for toggling icon
  String filePath = '';
  String jsonPath = '';
  double frequency = 0.0;
  double amplitude = 0.0;
  double decibel = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeRecorder();
  }

  Future<void> _initializeRecorder() async {
    await _recorder.openRecorder();
  }

  @override
  void dispose() {
    _recorder.closeRecorder();
    super.dispose();
  }

  Future<void> _startRecording() async {
    try {
      // Dapatkan direktori penyimpanan aplikasi
      Directory? appDir = await getExternalStorageDirectory();
      filePath = '${appDir?.path}/audio_record.aac';

      // Memeriksa dan meminta izin mikrofon
      var status = await Permission.microphone.request();
      if (!status.isGranted) {
        print("Microphone permission is not granted");
        return; // Jika izin tidak diberikan, tidak melanjutkan perekaman
      }

      // Mulai perekaman dan ubah status isRecording
      setState(() {
        isRecording = true;
      });

      // Mulai perekaman menggunakan flutter_sound recorder
      await _recorder.startRecorder(
        toFile: filePath,
        codec: Codec.aacADTS,
      );
      print("Recording started: $filePath");

      // Rekam selama 5 detik
      Timer(Duration(seconds: 3), () async {
        await _stopRecording();
      });
    } catch (e) {
      print("Error starting recorder: $e");
    }
  }

  Future<void> _stopRecording() async {
    await _recorder.stopRecorder();

    await verifyRecording(filePath);
    await verifyJsonFile(jsonPath);
    // print('Frequency: Hz');
    // print('Amplitude: ');
    // print('Decibel:  dB');
    setState(() {
      isRecording = false;
    });

    // Extract audio data
    File recordedFile = File(filePath);
    Uint8List audioData = await recordedFile.readAsBytes();
    Map<String, dynamic> audioInfo = await extractAudioData(audioData);
    // Save to JSON file
    await saveAudioDataToJson(audioInfo);
    setState(() {
      frequency = audioInfo['frequency'];
      amplitude = audioInfo['amplitude'];
      decibel = audioInfo['decibel'];
    });
  }
  Future<void> verifyRecording(String filePath) async {
    File audioFile = File(filePath);

    // Cek apakah file ada
    if (await audioFile.exists()) {
      // Cek ukuran file
      int fileSize = await audioFile.length();

      if (fileSize > 0) {
        print('Recording successful. File saved at $filePath with size $fileSize bytes');
      } else {
        print('Recording failed. The file is empty.');
      }
    } else {
      print('Recording failed. No file found at $filePath.');
    }
  }


  Future<void> verifyJsonFile(String jsonPath) async {
    File jsonFile = File(jsonPath);

    // Cek apakah file JSON ada
    if (await jsonFile.exists()) {
      // Cek ukuran file
      int fileSize = await jsonFile.length();

      if (fileSize > 0) {
        // Membaca isi file JSON untuk verifikasi tambahan
        String jsonData = await jsonFile.readAsString();
        print('JSON file saved successfully. Path: $jsonPath');
        print('File size: $fileSize bytes');
        print('File content: $jsonData');
      } else {
        print('JSON file exists, but it is empty.');
      }
    } else {
      print('Failed to save JSON file. No file found at $jsonPath.');
    }
  }


  Future<Map<String, dynamic>> extractAudioData(Uint8List audioData) async {
    // Konversi Uint8List ke List<int> (asumsi 16-bit PCM)
    List<int> audioListInt = [];
    for (int i = 0; i < audioData.length; i += 2) {
      int value = (audioData[i + 1] << 8) | audioData[i];
      if (value >= 0x8000) value -= 0x10000;
      audioListInt.add(value);
    }

    // Konversi ke List<double>
    List<double> audioList = audioListInt.map((e) => e.toDouble()).toList();

    // Hilangkan komponen DC (mean removal)
    double mean = audioList.reduce((a, b) => a + b) / audioList.length;
    audioList = audioList.map((v) => v - mean).toList();

    // Normalisasi amplitudo
    double maxAbsVal = audioList.map((v) => v.abs()).reduce((a, b) => a > b ? a : b);
    if (maxAbsVal > 0) {
      audioList = audioList.map((v) => v / maxAbsVal).toList();
    }

    // FFT
    final fft = FFT(audioList.length);
    final freqs = fft.realFft(audioList);

    // Hitung magnitudo spektral (menggabungkan komponen real dan imajiner)
    List<double> freqsDouble = freqs.map((f) => sqrt(f.x * f.x + f.y * f.y)).toList();
    double maxAmplitude = freqsDouble.reduce((curr, next) => curr.abs() > next.abs() ? curr : next);
    int maxFreqIndex = freqsDouble.indexOf(maxAmplitude);

    // Hitung frekuensi dominan
    double dominantFreq = (maxFreqIndex * 44100) / (audioList.length / 2); // Bagi dengan 2 karena simetri FFT

    // Batasi nilai amplitude minimum agar tidak nol atau terlalu kecil
    double minAmplitude = 1e-10;
    maxAmplitude = maxAmplitude.abs();
    if (maxAmplitude < minAmplitude) {
      maxAmplitude = minAmplitude;
    }

    // Hitung decibel
    double decibel = 20 * log(maxAmplitude) / log(10);

    return {
      'frequency': dominantFreq,
      'amplitude': maxAmplitude,
      'decibel': decibel,
    };
  }

  Future<void> saveAudioDataToJson(Map<String, dynamic> audioData) async {
    Directory? appDir = await getExternalStorageDirectory();
    jsonPath = '${appDir?.path}/audio_data_tolong.json';
    File jsonFile = File(jsonPath);

    String jsonString = jsonEncode(audioData);
    await jsonFile.writeAsString(jsonString);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Audio Recorder'),
        backgroundColor: Colors.brown[200],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            GestureDetector(
              onTap: isRecording ? null : _startRecording,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: isRecording ? Colors.red : Colors.brown[400],
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  isRecording ? Icons.mic : Icons.mic_off,
                  color: Colors.white,
                  size: 40,
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              'Frequency: ${frequency.toStringAsFixed(2)} Hz',
              style: TextStyle(fontSize: 18),
            ),
            Text(
              'Amplitude: ${amplitude.toStringAsFixed(2)}',
              style: TextStyle(fontSize: 18),
            ),
            Text(
              'Decibel: ${decibel.toStringAsFixed(2)} dB',
              style: TextStyle(fontSize: 18),
            ),
          ],
        ),
      ),
    );
  }
}