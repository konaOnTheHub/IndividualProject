import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_app/main.dart';

void main() {
  testWidgets('App builds and reaches health-check or error state', (WidgetTester tester) async {
    await tester.pumpWidget(const WasteClassifierApp());
    expect(find.text('Waste classifier'), findsOneWidget);

    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    final hasProgress = find.byType(CircularProgressIndicator).evaluate().isNotEmpty;
    final hasRetry = find.text('Retry').evaluate().isNotEmpty;
    final hasBoroughPrompt = find.textContaining('London borough').evaluate().isNotEmpty;

    expect(hasProgress || hasRetry || hasBoroughPrompt, isTrue);
  });
}
