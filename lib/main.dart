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
      Directory appDir = await getApplicationDocumentsDirectory();
      filePath = '${appDir.path}/audio_record.aac';

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
      Timer(Duration(seconds: 5), () async {
        await _stopRecording();
      });
    } catch (e) {
      print("Error starting recorder: $e");
    }
  }

  Future<void> _stopRecording() async {
    await _recorder.stopRecorder();
    print("file saved: $jsonPath");
    print("Recording started: $filePath");
    // print('Frequency: Hz');
    // print('Amplitude: ');
    // print('Decibel:  dB');
    setState(() {
      isRecording = false;
    });

    // Setelah rekaman selesai, ekstrak data audio
    File recordedFile = File(filePath);
    Uint8List audioData = await recordedFile.readAsBytes();
    Map<String, dynamic> audioInfo = await extractAudioData(audioData);

    // Simpan ke dalam JSON
    await saveAudioDataToJson(audioInfo);
  }

  Future<Map<String, dynamic>> extractAudioData(Uint8List audioData) async {
    List<double> audioList = audioData.map((byte) => byte.toDouble()).toList();
    final fft = FFT(audioList.length);
    final freqs = fft.realFft(audioList);

    List<double> freqsDouble = freqs.map((f) => f.x).toList();
    double maxAmplitude = freqsDouble.reduce((curr, next) => curr.abs() > next.abs() ? curr : next);
    int maxFreqIndex = freqsDouble.indexOf(maxAmplitude);
    double dominantFreq = maxFreqIndex * (44100 / audioList.length);

    double decibel = 20 * (log(maxAmplitude) / log(10));

    return {
      'frequency': dominantFreq,
      'amplitude': maxAmplitude,
      'decibel': decibel,
    };
  }

  Future<void> saveAudioDataToJson(Map<String, dynamic> audioData) async {
    Directory appDir = await getApplicationDocumentsDirectory();
    jsonPath = '${appDir.path}/audio_data_tolong.json';
    File jsonFile = File(jsonPath);

    String jsonString = jsonEncode(audioData);
    await jsonFile.writeAsString(jsonString);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Audio Recorder'),
      ),
      body: Center(
        child: GestureDetector(
          onTap: isRecording ? null : _startRecording,
          child: Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: isRecording ? Colors.red : Colors.brown,
              shape: BoxShape.circle,
            ),
            child: Icon(
              isRecording ? Icons.mic : Icons.mic_off,
              color: Colors.white,
              size: 40,
            ),
          ),
        ),
      ),
    );
  }
}