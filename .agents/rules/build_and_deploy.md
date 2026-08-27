# Workspace Rules: Build Verification & Deployment

## 1. Mandatory Build Verification
Whenever you make any code modifications or additions in this project:
- Always run `xcodebuild` (or `swiftc -typecheck`) to verify that the project builds cleanly with 0 errors before reporting completion.
- Command:
  ```bash
  xcodebuild build -project FaceReportDemo.xcodeproj -scheme FaceReportDemo -destination 'generic/platform=iOS Simulator'
  ```

## 2. Automatic Build & Deployment
- After confirming code correctness, automatically trigger a build so the latest artifacts are built and ready for deployment onto the user's simulator/device.
- If a running simulator is active, ensure the app bundle is updated or installed.
