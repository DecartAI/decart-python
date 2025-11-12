#!/usr/bin/env python3
"""
Release automation script for decart-python SDK.

This script automates the release process by:
1. Reading the current version from pyproject.toml
2. Prompting for the new version
3. Updating pyproject.toml
4. Committing and pushing changes
5. Creating a GitHub release
"""

import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def get_current_version() -> str:
    """Extract the current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found!")
        sys.exit(1)

    content = pyproject_path.read_text()
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        print("Error: Could not find version in pyproject.toml!")
        sys.exit(1)

    return match.group(1)


def update_version(new_version: str) -> None:
    """Update the version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Replace the version line
    updated_content = re.sub(
        r'^version = "[^"]+"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )

    pyproject_path.write_text(updated_content)
    print(f"✓ Updated version in pyproject.toml to {new_version}")


def validate_version(version: str) -> bool:
    """Validate version format (e.g., 0.0.4 or 1.2.3)."""
    return bool(re.match(r"^\d+\.\d+\.\d+$", version))


def get_release_notes() -> str:
    """Prompt user for release notes."""
    print("\nEnter release notes (press Ctrl+D or Ctrl+Z when done):")
    print("Example format:")
    print("## What's Changed")
    print("- Feature 1")
    print("- Bug fix 2")
    print("\n---")

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    return "\n".join(lines)


def main():
    """Main release process."""
    print("=" * 60)
    print("Decart Python SDK - Release Automation")
    print("=" * 60)

    # Get current version
    current_version = get_current_version()
    print(f"\nCurrent version: {current_version}")

    # Prompt for new version
    while True:
        new_version = input("\nEnter new version (e.g., 0.0.4): ").strip()
        if validate_version(new_version):
            break
        print("Invalid version format! Use format: X.Y.Z (e.g., 0.0.4)")

    # Confirm
    print(f"\n{current_version} → {new_version}")
    confirm = input("Continue with release? (y/N): ").strip().lower()
    if confirm != "y":
        print("Release cancelled.")
        sys.exit(0)

    # Update version in pyproject.toml
    update_version(new_version)

    # Commit and push
    print("\n" + "=" * 60)
    print("Git operations")
    print("=" * 60)

    run_command(["git", "add", "pyproject.toml"])
    run_command(["git", "commit", "-m", f"chore: bump version to {new_version}"])

    push_confirm = input("\nPush to remote? (y/N): ").strip().lower()
    if push_confirm == "y":
        try:
            run_command(["git", "push"])
            print("✓ Pushed to remote")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to push: {e.stderr.strip()}")
            print("⚠ You'll need to push manually later with: git push")
            continue_anyway = input("\nContinue with release anyway? (y/N): ").strip().lower()
            if continue_anyway != "y":
                print("Release cancelled.")
                sys.exit(1)
    else:
        print("⚠ Skipped push - remember to push manually!")

    # Create GitHub release
    print("\n" + "=" * 60)
    print("GitHub Release")
    print("=" * 60)

    create_release = input("\nCreate GitHub release? (y/N): ").strip().lower()
    if create_release != "y":
        print("\nRelease process completed (without GitHub release).")
        print("To create the release manually, run:")
        print(f'gh release create v{new_version} --title "v{new_version}" --notes "..."')
        sys.exit(0)

    # Get release title
    title = input(f"\nRelease title (default: v{new_version} - Release): ").strip()
    if not title:
        title = f"v{new_version} - Release"

    # Get release notes
    notes = get_release_notes()
    if not notes.strip():
        notes = f"## What's Changed\n- Version bump to {new_version}"

    # Create the release
    try:
        cmd = ["gh", "release", "create", f"v{new_version}", "--title", title, "--notes", notes]
        run_command(cmd)
        print(f"\n✓ Created GitHub release v{new_version}")
        print("✓ GitHub Actions will automatically publish to PyPI")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to create GitHub release: {e.stderr}")
        print("\nYou can create the release manually with:")
        print(f'gh release create v{new_version} --title "{title}" --notes "..."')
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Release process completed successfully!")
    print("=" * 60)
    print(f"\nVersion {new_version} has been released.")
    print("Monitor the GitHub Actions workflow for PyPI publication status.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRelease cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
