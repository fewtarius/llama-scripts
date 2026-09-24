#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Rebuild Script - Downloads ROCm SDK + Builds llama.cpp
# =============================================================================
# Handles full setup from a fresh checkout:
#   Linux + AMD:  Vulkan (default) or ROCm/HIP - downloads ROCm SDK
#   macOS (Apple Silicon): Metal backend - uses Xcode toolchain
#
# Usage:
#   ./scripts/rebuild.sh             # Default backend for this platform
#   ./scripts/rebuild.sh --rebuild   # Full rebuild (wipe deps/, re-download SDK)
#   ./scripts/rebuild.sh --rocm      # Build ROCm backend only (Linux)
#   ./scripts/rebuild.sh --vulkan    # Build Vulkan backend only (Linux)
#   ./scripts/rebuild.sh --metal     # Build Metal backend (macOS)
#   ./scripts/rebuild.sh --both      # Build both ROCm and Vulkan (Linux)
#   ./scripts/rebuild.sh --help      # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$PROJECT_ROOT/llama.cpp"
DEPS_DIR="$PROJECT_ROOT/deps"

# AMD stable ROCm pip repository
ROCM_PIP_INDEX="https://stable.repo.amd.com/rocm/whl-next/"

# ROCm SDK package version (stable)
ROCM_SDK_VERSION="${ROCM_SDK_VERSION:-10.0.0}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_ok()   { printf '%b[OK]%b   %s\n' "$GREEN" "$NC" "$1"; }
log_warn() { printf '%b[WARN]%b %s\n' "$YELLOW" "$NC" "$1"; }
log_error(){ printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1"; }

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build llama.cpp with the appropriate backend for the host platform.
  - Linux + AMD: downloads ROCm SDK, builds ROCm and/or Vulkan
  - macOS (Apple Silicon): builds Metal backend

OPTIONS:
    --rocm          Build ROCm/HIP backend only (Linux + AMD)
    --vulkan        Build Vulkan backend only (Linux + AMD)
    --metal         Build Metal backend (macOS)
    --both          Build both ROCm and Vulkan (Linux, default)
    --rebuild       Full rebuild (wipe deps/, re-download SDK)
    --clean         Clean build directories only (keep deps)
    -h, --help      Show this help

ENVIRONMENT:
    ROCM_SDK_VERSION  Stable ROCm SDK version (default: $ROCM_SDK_VERSION)
    ROCM_SDK_VERSION  Stable ROCm SDK version (default: $ROCM_SDK_VERSION)

EXAMPLES:
    $(basename "$0")                    # Build default backend for this platform
    $(basename "$0") --rebuild          # Full rebuild from scratch
    $(basename "$0") --rocm             # Build ROCm only (Linux)
    $(basename "$0") --vulkan           # Build Vulkan only (Linux)
    $(basename "$0") --metal            # Build Metal (macOS)

EOF
    exit 0
}

# =============================================================================
# Parse arguments and set platform-appropriate defaults
# =============================================================================

# Platform defaults: macOS -> Metal only; Linux -> Vulkan only.
case "$(uname -s)" in
    Darwin)
        BUILD_ROCM=false
        BUILD_VULKAN=false
        BUILD_METAL=true
        ;;
    *)
        BUILD_ROCM=false
        BUILD_VULKAN=true
        BUILD_METAL=false
        ;;
esac

CLEAN=false
REBUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm)
            BUILD_VULKAN=false
            BUILD_METAL=false
            BUILD_ROCM=true
            shift
            ;;
        --vulkan)
            BUILD_ROCM=false
            BUILD_METAL=false
            BUILD_VULKAN=true
            shift
            ;;
        --metal)
            BUILD_ROCM=false
            BUILD_VULKAN=false
            BUILD_METAL=true
            shift
            ;;
        --both)
            BUILD_ROCM=true
            BUILD_VULKAN=true
            BUILD_METAL=false
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Prerequisite checks
# =============================================================================

check_prereqs() {
    local missing=()

    # --- Build tools ---
    # cmake: can be system-installed, pip-installed, or bundled with ROCm LLVM
    local has_cmake=false
    for cmake_bin in cmake "$HOME/.local/bin/cmake" "$PROJECT_ROOT/deps/lib/llvm/bin/cmake"; do
        if command -v "$cmake_bin" &>/dev/null; then
            has_cmake=true
            break
        fi
    done
    if [[ "$has_cmake" == false ]]; then
        missing+=("cmake")
    fi

    # make: required by cmake's Makefile generator (Linux only)
    # macOS uses ninja or the default Xcode generator; ninja is checked later
    if [[ "$(uname -s)" != "Darwin" ]]; then
        if ! command -v make &>/dev/null; then
            missing+=("make")
        fi
    fi

    # curl, git: used for downloading and submodule management
    for tool in curl git; do
        if ! command -v "$tool" &>/dev/null; then
            missing+=("$tool")
        fi
    done

    # --- C/C++ compiler ---
    # Priority: ROCm bundled clang > system clang > system gcc (Linux)
    # On macOS, prefer Apple's clang (from Xcode Command Line Tools)
    local has_compiler=false
    for compiler in clang++ g++; do
        if command -v "$compiler" &>/dev/null; then
            has_compiler=true
            break
        fi
    done
    if [[ "$has_compiler" == false ]]; then
        missing+=("clang++ or g++ (C++ compiler)")
    fi

    # --- GCC runtime libraries (Linux ROCm builds) ---
    # The ROCm SDK's bundled clang links against system libgcc/libstdc++.
    # On minimal distros (SteamOS, etc.) GCC may not be installed.
    if [[ "$(uname -s)" != "Darwin" ]]; then
        if ! command -v gcc &>/dev/null; then
            # Check if libgcc is available (some distros ship libgcc_s without gcc)
            if ! ldconfig -p 2>/dev/null | grep -q libgcc; then
                missing+=("gcc (provides libgcc/libstdc++ needed by ROCm clang linker)")
            fi
        fi
    fi

   # --- Vulkan-specific ---
    if [[ "$BUILD_VULKAN" == true ]]; then
        # Vulkan shader compilation needs glslc or glslangValidator
        if ! command -v glslc &>/dev/null && ! command -v glslangValidator &>/dev/null; then
            missing+=("glslc (vulkan-shaders)")
        fi
    fi

    # --- Metal-specific (macOS) ---
    if [[ "$BUILD_METAL" == true ]]; then
        if ! command -v xcrun &>/dev/null; then
            missing+=("xcrun (Xcode Command Line Tools)")
        fi
    fi

    # Report any missing prerequisites
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for m in "${missing[@]}"; do
            echo "  - $m"
        done
        if [[ "$(uname -s)" == "Darwin" ]]; then
            echo ""
            echo "Install Xcode Command Line Tools:"
            echo "  xcode-select --install"
        fi
        exit 1
    fi
}

# =============================================================================
# GPU detection
# =============================================================================

source "$SCRIPT_DIR/detect-gpu.sh"

# Show the detected platform/GPU
if [[ "$(uname -s)" == "Darwin" ]]; then
    log_info "Platform: macOS (${LLAMA_GPU_NAME:-unknown})"
    log_info "Metal acceleration enabled via Apple Silicon GPU"
else
    if [[ -z "$LLAMA_ROCM_VARIANT" ]]; then
        log_warn "GPU not in detection map. Defaulting to gfx110X."
        log_warn "Set LLAMA_ROCM_VARIANT in your environment if this is wrong."
        LLAMA_ROCM_VARIANT="gfx110X"
    fi
    log_info "GPU: ${LLAMA_GPU_NAME:-unknown} ($LLAMA_GPU_PCI_ID)"
    log_info "GFX: $LLAMA_GFX_ARCH ($LLAMA_GFX_VERSION)"
    log_info "SDK variant: $LLAMA_ROCM_VARIANT"
fi

# =============================================================================
# Download ROCm SDK (Linux only)
# =============================================================================

download_rocm() {
    if [[ "$BUILD_ROCM" != true ]]; then
        return 0
    fi

    # Check if already extracted
    if [[ -f "$DEPS_DIR/lib/libamdhip64.so" ]] && [[ -f "$DEPS_DIR/lib/llvm/bin/clang" ]] && [[ "$REBUILD" != true ]]; then
        log_ok "ROCm SDK already installed at $DEPS_DIR"
        return 0
    fi

    # Full rebuild: wipe deps
    if [[ "$REBUILD" == true ]] && [[ -d "$DEPS_DIR" ]]; then
        log_info "Removing existing deps for full rebuild..."
        rm -rf "$DEPS_DIR"
    fi

    local rocm_ver="$ROCM_SDK_VERSION"
    local pip_repo="$ROCM_PIP_INDEX"

    # Map llama-scripts GPU detection to ROCm pip device packages.
    # LLAMA_ROCM_VARIANT is a tarball variant (e.g. gfx120X) used for nightly
    # therock tarballs. For stable pip packages we need the actual architecture
    # (e.g. gfx1151) or an -all- variant for combined architectures.
    local device_pkg=""
    case "$LLAMA_GFX_ARCH" in
        gfx90*)
            device_pkg="rocm-sdk-device-gfx90a"
            ;;
        gfx101*)
            device_pkg="rocm-sdk-device-gfx1012"
            ;;
        gfx103*)
            device_pkg="rocm-sdk-device-gfx1030"
            ;;
        gfx110*)
            device_pkg="rocm-sdk-device-gfx110X-all"
            ;;
        gfx115*)
            device_pkg="rocm-sdk-device-${LLAMA_GFX_ARCH}"
            ;;
        gfx120*)
            device_pkg="rocm-sdk-device-${LLAMA_GFX_ARCH}"
            ;;
        *)
            # Default to broad -all- variant for combined architecture coverage
            device_pkg="rocm-sdk-device-gfx110X-all"
            ;;
    esac

    # Packages to download (core, libraries, device-specific, devel headers/cmake)
    local -a pkgs=(
        "rocm-sdk-core==${rocm_ver}"
        "rocm-sdk-libraries==${rocm_ver}"
        "${device_pkg}==${rocm_ver}"
        "rocm-sdk-devel==${rocm_ver}"
    )

    local tmp_wheel_dir
    tmp_wheel_dir=$(mktemp -d)
    trap "rm -rf '$tmp_wheel_dir'" RETURN

    log_info "Downloading ROCm SDK ${rocm_ver} (stable) from AMD pip repo..."
    log_info "Packages: ${pkgs[*]}"
    log_info "Device variant: $device_pkg"
    log_info "This is ~2.6 GB total (may take several minutes...)"

    # Download all wheels
    if ! pip3 download --no-deps -d "$tmp_wheel_dir" "${pkgs[@]}" --index-url "$pip_repo" 2>&1 | tail -5; then
        log_error "Failed to download ROCm SDK packages"
        log_error "Try: pip3 download --no-deps -d /tmp/rocm \"${pkgs[*]}\" --index-url $pip_repo"
        exit 1
    fi

    mkdir -p "$DEPS_DIR"

    # --- Extract rocm-sdk-core (biggest, contains compiler + runtime + headers) ---
    local core_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_core-*.whl 2>/dev/null | head -1)
    if [[ -z "$core_wheel" ]]; then
        log_error "rocm-sdk-core wheel not found"
        exit 1
    fi
    log_info "Extracting rocm-sdk-core..."
    python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$core_wheel')
prefix = None
for name in z.namelist():
    if name.startswith('_rocm_sdk_core/'):
        prefix = '_rocm_sdk_core/'
        break
if not prefix:
    print('ERROR: _rocm_sdk_core/ prefix not found in wheel')
    exit(1)
dest_root = '$DEPS_DIR'
count = 0
for name in z.namelist():
    if name.startswith(prefix) and not name.endswith('/'):
        relpath = name[len(prefix):]
        dest = os.path.join(dest_root, relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        count += 1
print(f'  Extracted {count} files')
"

    # --- Extract rocm-sdk-libraries (hipBLAS, rocBLAS, MIOpen, etc.) ---
    local libs_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_libraries-*.whl 2>/dev/null | head -1)
    if [[ -n "$libs_wheel" ]]; then
        log_info "Extracting rocm-sdk-libraries..."
        python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$libs_wheel')
prefix = '_rocm_sdk_libraries/'
count = 0
for name in z.namelist():
    if name.startswith(prefix) and not name.endswith('/'):
        relpath = name[len(prefix):]
        dest = os.path.join('$DEPS_DIR', relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        count += 1
print(f'  Extracted {count} files')
"
    fi

    # --- Extract device-specific code objects ---
    local device_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_device_*-$rocm_ver-*.whl 2>/dev/null | head -1)
    if [[ -n "$device_wheel" ]]; then
        log_info "Extracting device code for $LLAMA_ROCM_VARIANT..."
        python3 -c "
import zipfile, os, shutil
z = zipfile.ZipFile('$device_wheel')
# Device wheels may have different top-level prefixes
prefixes = set()
for name in z.namelist():
    if name and not name.endswith('/'):
        top = name.split('/')[0]
        if top.startswith('_'):
            prefixes.add(top + '/')

count = 0
for name in z.namelist():
    if name.endswith('/'):
        continue
    relpath = name
    for p in prefixes:
        if name.startswith(p):
            relpath = name[len(p):]
            break
    dest = os.path.join('$DEPS_DIR', relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with z.open(name) as src, open(dest, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    count += 1
print(f'  Extracted {count} files')
"
    fi

    # --- Extract devel package (cmake config files, additional headers) ---
    local devel_wheel=$(ls "$tmp_wheel_dir"/rocm_sdk_devel-*.whl 2>/dev/null | head -1)
    if [[ -n "$devel_wheel" ]]; then
        log_info "Extracting rocm-sdk-devel (cmake configs)..."
        python3 -c "
import zipfile, io, tarfile, os, shutil
z = zipfile.ZipFile('$devel_wheel')
# The devel package contains an internal _devel.tar that holds the actual files
tar_members = [n for n in z.namelist() if n.endswith('_devel.tar')]
for tar_name in tar_members:
    data = z.read(tar_name)
    t = tarfile.open(fileobj=io.BytesIO(data))
    # Determine the prefix from the first member
    sample = t.getmembers()[0] if t.getmembers() else None
    prefix = '_rocm_sdk_devel/'
    count = 0
    for m in t.getmembers():
        if not m.isfile():
            continue
        if m.name.startswith(prefix):
            relpath = m.name[len(prefix):]
        else:
            relpath = m.name
        dest = os.path.join('$DEPS_DIR', relpath)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f = t.extractfile(m)
        if f:
            shutil.copyfileobj(f, open(dest, 'wb'))
        count += 1
    print(f'  Extracted {count} files from {tar_name}')
"
    fi

    # --- Create symlinks for versioned library names ---
    # The stable pip packages ship versioned .so files (e.g. libamdhip64.so.7)
    # but not the unversioned .so symlink. The cmake config files reference
    # specific versioned filenames. Create symlinks so all references resolve.
    log_info "Creating compatibility symlinks..."
    _create_rocm_symlinks "$DEPS_DIR/lib"

    # Clean up temporary wheel directory
    rm -rf "$tmp_wheel_dir"

    # Fix permissions on executables
    find "$DEPS_DIR" -type f \( -name "clang*" -o -name "hipcc" -o -name "hipconfig" -o -name "rocm-*" \) -exec chmod +x {} + 2>/dev/null || true
    find "$DEPS_DIR/lib/llvm/bin" -type f -exec chmod +x {} + 2>/dev/null || true

    # Verify
    if [[ -f "$DEPS_DIR/lib/libamdhip64.so" || -f "$DEPS_DIR/lib/libamdhip64.so.7" ]]; then
        log_ok "ROCm ${rocm_ver} SDK installed to $DEPS_DIR"
        log_info "Libraries: $(ls "$DEPS_DIR/lib/"*.so 2>/dev/null | wc -l) shared objects"
        
        # Register library paths with ldconfig so binaries can find them
        # without needing LD_LIBRARY_PATH sourced
        local ldconf="/etc/ld.so.conf.d/llama-scripts-rocm.conf"
        if sudo sh -c "cat > '$ldconf'" 2>/dev/null <<EOF
$DEPS_DIR/lib/llvm/lib
$DEPS_DIR/lib
$DEPS_DIR/lib/rocm_sysdeps/lib
EOF
        then
            sudo ldconfig 2>/dev/null || true
            log_info "Registered ROCm library paths with ldconfig"
        else
            log_warn "Could not register ROCm paths with ldconfig (non-root) - source env.sh before running"
        fi
    else
        log_error "ROCm SDK extraction failed - libamdhip64.so not found"
        exit 1
    fi

    # Verify ROCm bundled tools are accessible
    local rocm_clang="$DEPS_DIR/lib/llvm/bin/clang"
    if [[ ! -x "$rocm_clang" ]]; then
        log_error "ROCm clang not found at $rocm_clang"
        log_error "The SDK may be incomplete or corrupted. Try --rebuild."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Create symlinks so cmake config files (which reference specific versioned .so
# filenames) can find the libraries shipped by the stable ROCm packages.
# -----------------------------------------------------------------------------
_create_rocm_symlinks() {
    local lib_dir="$1"
    [[ -d "$lib_dir" ]] || return 0
    cd "$lib_dir" || return 0

    # Create unversioned .so symlinks from .so.MAJOR files (or shortest version)
    # Do this first so the Python step below can use them as fallback candidates.
    for f in *.so.*; do
        local base
        base=$(echo "$f" | sed 's/\.so\..*//')
        if [[ "$base" != "$f" ]] && [[ ! -L "${base}.so" ]] && [[ ! -f "${base}.so" ]]; then
            ln -sf "$f" "${base}.so" 2>/dev/null || true
        fi
    done

    # For each cmake targets file, find IMPORTED_LOCATION_RELEASE entries and
    # create symlinks from the referenced versioned name to the actual library
    if command -v python3 &>/dev/null; then
        python3 -c "
import os, re, glob

base = '$lib_dir'

# Collect all referenced library names from cmake targets files
referenced = set()
for cmake_path in glob.glob(os.path.join(base, 'cmake', '**', '*.cmake'), recursive=True):
    try:
        with open(cmake_path) as f:
            content = f.read()
        for m in re.finditer(r'lib([a-zA-Z0-9_-]+)\.so\.([0-9][0-9.\-]*)', content):
            referenced.add(m.group(0))
    except:
        pass

created = 0
for ref in sorted(referenced):
    if os.path.exists(ref):
        continue
    # Find the base library name (strip version entirely)
    base_name = ref.split('.so')[0] + '.so'
    # Build candidate paths by progressively stripping version components
    # e.g. for libhsa-runtime64.so.1.21.0:
    #   candidates = ['libhsa-runtime64.so', 'libhsa-runtime64.so.1',
    #                 'libhsa-runtime64.so.1.21', 'libhsa-runtime64.so.1.21.0']
    # We try longest version suffix first (most specific match)
    vparts = ref.split('.so.')[1].split('.') if '.so.' in ref else []
    candidates = [base_name]
    for i in range(1, len(vparts) + 1):
        candidates.append(base_name + '.' + '.'.join(vparts[:i]))
    # Reverse so we try longest match first
    candidates.reverse()

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break
    if found:
        os.symlink(found, ref)
        created += 1
    else:
        # Try any matching .so file with the same library prefix
        for f in os.listdir('.'):
            if f.startswith(base_name) and '.so' in f and f != ref:
                if os.path.islink(f) or os.path.isfile(f):
                    os.symlink(f, ref)
                    created += 1
                    break

print(f'  Created {created} compatibility symlinks')
" || true
    fi

    # Create unversioned .so symlinks from .so.MAJOR files (or shortest version)
    for f in *.so.*; do
        local base
        base=$(echo "$f" | sed 's/\.so\..*//')
        if [[ "$base" != "$f" ]] && [[ ! -L "${base}.so" ]] && [[ ! -f "${base}.so" ]]; then
            ln -sf "$f" "${base}.so" 2>/dev/null || true
        fi
    done

    cd - >/dev/null 2>&1 || true
}

# =============================================================================
# Initialize submodule
# =============================================================================

init_submodule() {
    if [[ ! -f "$LLAMA_DIR/CMakeLists.txt" ]]; then
        log_info "Initializing llama.cpp submodule..."
        cd "$PROJECT_ROOT"
        git submodule update --init --recursive
    fi
}

# =============================================================================
# Patch management
# =============================================================================

apply_patches() {
    local patch_dir="$PROJECT_ROOT/patches"

    if [[ ! -d "$patch_dir" ]]; then
        return 0
    fi

    for patch_file in "$patch_dir"/*.patch; do
        [[ -f "$patch_file" ]] || continue

        local patch_name
        patch_name=$(basename "$patch_file")

        # Check if patch is already applied using git apply --check
        # If --check fails, the patch is already applied or conflicts
        if ! git -C "$LLAMA_DIR" apply --check "$patch_file" 2>/dev/null; then
            log_info "Patch already applied or not applicable: $patch_name"
            continue
        fi

        log_info "Applying patch: $patch_name"
        git -C "$LLAMA_DIR" apply "$patch_file"
        log_ok "Applied: $patch_name"
    done
}

# =============================================================================
# Build functions
# =============================================================================

build_rocm() {
    log_info "Building ROCm backend..."

    local build_dir="$PROJECT_ROOT/src/llama-rocm/build"

    # Clean if requested
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Apply patches to submodule
    apply_patches

    # Configure
    cd "$build_dir"

    # Use detected GFX architecture for HIP compilation target
    local hip_arch="${LLAMA_GFX_ARCH:-gfx1103}"
    log_info "HIP target architecture: $hip_arch"

    cmake "$LLAMA_DIR" \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
      -DCMAKE_HIP_PLATFORM=amd \
      -DCMAKE_HIP_ARCHITECTURES="$hip_arch" \
      -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH/lib/llvm" \
       -DGGML_HIP=ON \
        -DGGML_HIPBLAS=ON \
        -DGGML_HIP_NO_VMM=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_CPU=ON \
        -DGGML_NATIVE=OFF \
        $LLAMA_CMAKE_CPU_FLAGS \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON

    # Build
    cmake --build . --config Release -j$(nproc)

    log_ok "ROCm build complete: $build_dir/bin/llama-server"
    log_info "CPU ISA level: $LLAMA_CPU_ISA"
}

build_vulkan() {
    log_info "Building Vulkan backend..."

    local build_dir="$PROJECT_ROOT/src/llama-vulkan/build"

    # Clean if requested
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Apply patches to submodule
    apply_patches

    # Configure
    cd "$build_dir"
    cmake "$LLAMA_DIR" \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
      -DGGML_HIP=OFF \
       -DGGML_HIPBLAS=OFF \
        -DGGML_VULKAN=ON \
        -DGGML_CPU=ON \
        -DGGML_NATIVE=OFF \
        $LLAMA_CMAKE_CPU_FLAGS \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        2>&1 | tail -5

    # Build
    cmake --build . --config Release -j$(nproc)

    log_ok "Vulkan build complete: $build_dir/bin/llama-server"
    log_info "CPU ISA level: $LLAMA_CPU_ISA"
}

build_metal() {
    log_info "Building Metal backend (macOS)..."

    local build_dir="$PROJECT_ROOT/src/llama-metal/build"

    # Clean if requested
    [[ "$CLEAN" == true || "$REBUILD" == true ]] && rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Apply patches to submodule
    apply_patches

    # Configure
    cd "$build_dir"

    # Use Apple clang. GGML_METAL defaults to ON on Apple platforms in llama.cpp,
    # so we set it explicitly to be safe.
    cmake "$LLAMA_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DGGML_METAL=ON \
        -DGGML_METAL_NDEBUG=ON \
        -DGGML_HIP=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_CPU=ON \
        -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        2>&1 | tail -5

    # Build with the detected logical core count
    local jobs
    jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "$(nproc)")
    cmake --build . --config Release -j"$jobs"

    log_ok "Metal build complete: $build_dir/bin/llama-server"
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo -e "${CYAN}=== Llama.cpp Build Script ===${NC}"
echo -e "${CYAN}  Platform: $(uname -s) $(uname -m)${NC}"
echo -e "${CYAN}  GPU: ${LLAMA_GPU_NAME:-unknown} (${LLAMA_GFX_ARCH:-?})${NC}"
echo -e "${CYAN}  CPU ISA: ${LLAMA_CPU_ISA:-unknown}${NC}"
echo -e "${CYAN}  CMake CPU flags: ${LLAMA_CMAKE_CPU_FLAGS:-none}${NC}"
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${CYAN}  ROCm: ${ROCM_SDK_VERSION}${NC}"
fi
echo ""

# Step 1: Initialize submodule
init_submodule

# Step 2: Download ROCm SDK (Linux ROCm builds only)
download_rocm

# Step 3: Source environment so ROCm tools (clang, etc.) are on PATH (Linux only)
if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "$BUILD_ROCM" == true || "$BUILD_VULKAN" == true ]]; then
        if [[ "$BUILD_ROCM" == true ]]; then
            source "$PROJECT_ROOT/scripts/env.sh" rocm
        else
            # Vulkan builds still need ROCm's bundled clang/lld for compilation
            export PATH="$PROJECT_ROOT/deps/lib/llvm/bin:$PATH"
        fi
    fi
fi

# Step 4: Check prerequisites
check_prereqs

# Step 5: Build requested backends
[[ "$BUILD_ROCM" == true ]]   && build_rocm
[[ "$BUILD_VULKAN" == true ]] && build_vulkan
[[ "$BUILD_METAL" == true ]]  && build_metal

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "Binaries:"
[[ "$BUILD_ROCM" == true ]]   && echo "  ROCm:   $PROJECT_ROOT/src/llama-rocm/build/bin/llama-server"
[[ "$BUILD_VULKAN" == true ]] && echo "  Vulkan: $PROJECT_ROOT/src/llama-vulkan/build/bin/llama-server"
[[ "$BUILD_METAL" == true ]]  && echo "  Metal:  $PROJECT_ROOT/src/llama-metal/build/bin/llama-server"
echo ""
echo "Next: drop a GGUF model in models/ and run ./llama-run.sh --server"
echo ""
