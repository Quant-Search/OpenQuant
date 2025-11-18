#!/bin/bash
set -e

echo "🚀 Starting AppImage Creation..."

# 1. Build the binaries using PyInstaller
echo "🔨 Building binaries..."
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi
$PYTHON scripts/build_app.py

# 2. Prepare AppDir
echo "📂 Preparing AppDir..."
rm -rf AppDir
mkdir -p AppDir
cp -r dist/OpenQuantRobot/* AppDir/

# 3. Create Metadata
echo "📝 Creating metadata..."
cat > AppDir/OpenQuant.desktop <<EOF
[Desktop Entry]
Type=Application
Name=OpenQuant Robot
Exec=OpenQuantRobot
Icon=openquant
Categories=Finance;
Terminal=false
EOF

# Create a dummy icon if none exists
if [ ! -f AppDir/openquant.png ]; then
    touch AppDir/openquant.png
fi

# Create AppRun symlink
ln -s OpenQuantRobot AppDir/AppRun

# 4. Download AppImageTool if needed
if [ ! -f appimagetool-x86_64.AppImage ]; then
    echo "⬇️ Downloading appimagetool..."
    wget -q https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

# 5. Generate AppImage
echo "📦 Packaging AppImage..."
# ARCH=x86_64 ./appimagetool-x86_64.AppImage AppDir OpenQuantRobot-x86_64.AppImage
# Using --no-appstream to avoid validation errors on dummy metadata
ARCH=x86_64 ./appimagetool-x86_64.AppImage --no-appstream AppDir OpenQuantRobot-x86_64.AppImage

echo "✅ AppImage created: OpenQuantRobot-x86_64.AppImage"
echo "To run: ./OpenQuantRobot-x86_64.AppImage"
