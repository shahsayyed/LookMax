# Agent Rules for LookMax

1. **Verify iOS App Builds When App Code Changes**:
   Whenever changes are made to the iOS application codebase (under `iOS/`), run a build test using `xcodebuild` to ensure there are no build errors. Do not run `xcodebuild` for Python ML scripts or pipeline-only changes.

2. **Automatic Re-deployment**:
   Automatically trigger the iOS build when iOS code is modified so that the newest binaries are prepared for testing on the simulator or target device.
