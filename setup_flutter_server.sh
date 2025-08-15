#!/bin/bash
# Flutter/Dart server setup script

set -e

# Check for Flutter installation
echo "Checking for Flutter..."
if ! command -v flutter &> /dev/null; then
  echo "Flutter is not installed. Please install Flutter from https://flutter.dev/docs/get-started/install."
  exit 1
fi

echo "Flutter found: $(flutter --version | head -n 1)"


# Ensure we are in the aiux project root (where pubspec.json is)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "Running flutter pub get..."
flutter pub get

echo "Setup complete. To run your Flutter app, use:"
echo "  flutter run"
echo "Or for web:"
echo "  flutter run -d chrome"
