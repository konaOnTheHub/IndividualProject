import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

// --- API models (contract from FastAPI `HealthResponse` / `ClassifyResponse`) ---

class HealthStatus {
  const HealthStatus({
    required this.status,
    required this.device,
    required this.modelPath,
    required this.modelLoaded,
  });

  final String status;
  final String device;
  final String modelPath;
  final bool modelLoaded;

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'] as String,
      device: json['device'] as String,
      modelPath: json['model_path'] as String,
      modelLoaded: json['model_loaded'] as bool,
    );
  }

  bool get isServiceReady => status == 'ok' && modelLoaded;
}

class ClassificationResult {
  const ClassificationResult({
    required this.category,
    required this.confidence,
    required this.classIndex,
    required this.probabilities,
  });

  final String category;
  final double confidence;
  final int classIndex;
  final Map<String, double> probabilities;

  factory ClassificationResult.fromJson(Map<String, dynamic> json) {
    final raw = json['probabilities'] as Map<String, dynamic>;
    final probs = <String, double>{};
    for (final e in raw.entries) {
      probs[e.key] = (e.value as num).toDouble();
    }
    return ClassificationResult(
      category: json['category'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      classIndex: json['class_index'] as int,
      probabilities: probs,
    );
  }
}

// --- Confidence / top-K (same rules as prior `confidence_rules.dart`) ---

int topCountForConfidence(double confidence) {
  if (confidence >= 0.70) return 1;
  if (confidence >= 0.45) return 2;
  return 3;
}

List<MapEntry<String, double>> topCandidates(
  Map<String, double> probabilities,
  double confidence,
) {
  final entries = probabilities.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  return entries.take(topCountForConfidence(confidence)).toList();
}

// --- API client ---

String resolveBaseUrl() {
  const configured = String.fromEnvironment('API_BASE_URL', defaultValue: '');
  if (configured.isNotEmpty) return configured;
  if (Platform.isAndroid) return 'http://10.0.2.2:8000';
  return 'http://127.0.0.1:8000';
}

Future<HealthStatus> fetchHealth(String baseUrl) async {
  final response = await http.get(Uri.parse('$baseUrl/health'));
  if (response.statusCode != 200) {
    throw Exception('Health check failed (HTTP ${response.statusCode}).');
  }
  return HealthStatus.fromJson(
    jsonDecode(response.body) as Map<String, dynamic>,
  );
}

Future<ClassificationResult> classifyImage(String baseUrl, String imagePath) async {
  final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/classify'))
    ..files.add(await http.MultipartFile.fromPath('file', imagePath));
  final streamed = await request.send();
  final response = await http.Response.fromStream(streamed);

  if (response.statusCode == 200) {
    return ClassificationResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  if (response.statusCode == 400 ||
      response.statusCode == 413 ||
      response.statusCode == 415) {
    throw Exception(_parseApiDetail(response.body));
  }
  throw Exception('Classification failed (HTTP ${response.statusCode}).');
}

String _parseApiDetail(String body) {
  try {
    final payload = jsonDecode(body) as Map<String, dynamic>;
    return payload['detail'] as String? ?? 'Unexpected API error.';
  } catch (_) {
    return 'Unexpected API error.';
  }
}

// --- Borough guidance (loaded from asset JSON) ---

const kBoroughs = [
  'Barnet',
  'Camden',
  'Greenwich',
  'Hackney',
  'Lambeth',
];

Future<Map<String, Map<String, String>>> loadBoroughGuidance() async {
  final raw = await rootBundle.loadString('assets/borough_guidance.json');
  final decoded = jsonDecode(raw) as Map<String, dynamic>;
  final out = <String, Map<String, String>>{};
  for (final e in decoded.entries) {
    final inner = e.value as Map<String, dynamic>;
    out[e.key] = inner.map((k, v) => MapEntry(k, v as String));
  }
  return out;
}

String guidanceFor(
  Map<String, Map<String, String>>? guidance,
  String borough,
  String category,
) {
  final line = guidance?[borough]?[category];
  if (line != null && line.isNotEmpty) return line;
  return 'Check your council’s A–Z waste guide or reuse and recycling centre for how to dispose of this material safely.';
}

// --- App ---

void main() {
  runApp(const WasteClassifierApp());
}

class WasteClassifierApp extends StatelessWidget {
  const WasteClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Waste Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const WasteHomePage(),
    );
  }
}

class WasteHomePage extends StatefulWidget {
  const WasteHomePage({super.key});

  @override
  State<WasteHomePage> createState() => _WasteHomePageState();
}

class _WasteHomePageState extends State<WasteHomePage> {
  final String _baseUrl = resolveBaseUrl();
  final ImagePicker _picker = ImagePicker();

  bool _checkingHealth = true;
  String? _startupError;
  Map<String, Map<String, String>>? _guidance;
  String? _guidanceLoadError;

  String? _selectedBorough;
  bool _classifying = false;
  String? _classifyError;
  ClassificationResult? _result;

  @override
  void initState() {
    super.initState();
    _runStartupChecks();
  }

  Future<void> _runStartupChecks() async {
    setState(() {
      _checkingHealth = true;
      _startupError = null;
      _guidanceLoadError = null;
    });

    try {
      final healthFuture = fetchHealth(_baseUrl);
      final guidanceFuture = loadBoroughGuidance();
      final health = await healthFuture;
      if (!health.isServiceReady) {
        setState(() {
          _checkingHealth = false;
          _startupError =
              'The classification service is not ready (model loaded: ${health.modelLoaded}).';
        });
        return;
      }
      try {
        _guidance = await guidanceFuture;
      } catch (e) {
        _guidanceLoadError = 'Could not load borough guidance: $e';
        _guidance = {};
      }
      setState(() {
        _checkingHealth = false;
        _startupError = null;
      });
    } catch (e) {
      setState(() {
        _checkingHealth = false;
        _startupError = 'Cannot reach the API at $_baseUrl.\n$e';
      });
    }
  }

  Future<void> _takePhotoAndClassify() async {
    if (_selectedBorough == null) return;

    setState(() {
      _classifyError = null;
      _result = null;
    });

    final picked = await _picker.pickImage(
      source: ImageSource.camera,
      maxWidth: 2048,
      maxHeight: 2048,
      imageQuality: 88,
    );
    if (picked == null || !mounted) return;

    setState(() => _classifying = true);
    try {
      final result = await classifyImage(_baseUrl, picked.path);
      if (!mounted) return;
      setState(() {
        _classifying = false;
        _result = result;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _classifying = false;
        _classifyError = '$e';
      });
    }
  }

  void _resetResult() {
    setState(() {
      _result = null;
      _classifyError = null;
    });
  }

  void _resetBorough() {
    setState(() {
      _selectedBorough = null;
      _result = null;
      _classifyError = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Waste classifier'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_checkingHealth) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Checking API…'),
          ],
        ),
      );
    }

    if (_startupError != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.cloud_off, size: 56, color: Theme.of(context).colorScheme.error),
              const SizedBox(height: 16),
              Text(
                _startupError!,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 8),
              Text(
                'Base URL: $_baseUrl',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _runStartupChecks,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_selectedBorough == null) {
      return _buildBoroughStep(context);
    }

    if (_result != null) {
      return _buildResultStep(context);
    }

    return _buildCaptureStep(context);
  }

  Widget _buildBoroughStep(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        if (_guidanceLoadError != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: Material(
              color: Theme.of(context).colorScheme.errorContainer,
              borderRadius: BorderRadius.circular(12),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  _guidanceLoadError!,
                  style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
                ),
              ),
            ),
          ),
        Text(
          'Which London borough are you in?',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 8),
        Text(
          'We’ll show disposal tips for your area after classification.',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
        ),
        const SizedBox(height: 24),
        ...kBoroughs.map(
          (b) => Card(
            child: ListTile(
              title: Text(b),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => setState(() => _selectedBorough = b),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCaptureStep(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Borough: $_selectedBorough', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 24),
          Text(
            'Take a clear photo of the waste item. The image is sent to your classifier API.',
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: 24),
          if (_classifying)
            const Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Classifying…'),
                  ],
                ),
              ),
            )
          else ...[
            FilledButton.icon(
              onPressed: _takePhotoAndClassify,
              icon: const Icon(Icons.photo_camera),
              label: const Text('Take photo & classify'),
            ),
            if (_classifyError != null) ...[
              const SizedBox(height: 16),
              Text(
                _classifyError!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            const Spacer(),
            TextButton(
              onPressed: _resetBorough,
              child: const Text('Change borough'),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildResultStep(BuildContext context) {
    final result = _result!;
    final borough = _selectedBorough!;
    final candidates = topCandidates(result.probabilities, result.confidence);

    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Results', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 8),
        Text(
          topCountForConfidence(result.confidence) > 1
              ? 'The model is not fully sure — showing the top ${candidates.length} possibilities:'
              : 'Top prediction:',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
        ),
        const SizedBox(height: 16),
        ...candidates.asMap().entries.map((e) {
          final i = e.key;
          final c = e.value;
          final pct = (c.value * 100).toStringAsFixed(1);
          final guide = guidanceFor(_guidance, borough, c.key);
          return Card(
            margin: const EdgeInsets.only(bottom: 12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      if (i == 0)
                        Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: Icon(
                            Icons.star,
                            size: 20,
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ),
                      Expanded(
                        child: Text(
                          c.key,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                        ),
                      ),
                      Text('$pct%', style: Theme.of(context).textTheme.titleMedium),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(guide, style: Theme.of(context).textTheme.bodyMedium),
                ],
              ),
            ),
          );
        }),
        const SizedBox(height: 16),
        FilledButton(
          onPressed: _resetResult,
          child: const Text('Classify another item'),
        ),
        const SizedBox(height: 8),
        OutlinedButton(
          onPressed: _resetBorough,
          child: const Text('Change borough'),
        ),
      ],
    );
  }
}
