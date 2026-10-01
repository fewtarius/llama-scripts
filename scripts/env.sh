#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp Environment Setup
# =============================================================================
# Source this file to set up the environment for llama.cpp
#   source scripts/env.sh
#   source scripts/env.sh rocm   # or: source scripts/env.sh vulkan
#
# This sets up paths for the specified backend.

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse backend argument (default: rocm)
BACKEND="${1:-rocm}"
BACKEND=$(echo "$BACKEND" | tr "[:upper:]" "[:lower:]")  # lowercase (POSIX-safe)

# Validate backend
if [[ "$BACKEND" != "rocm" && "$BACKEND" != "vulkan" && "$BACKEND" != "metal" ]]; then
    echo "ERROR: Invalid backend '$BACKEND'. Use 'rocm', 'vulkan', or 'metal'"
    return 1 2>/dev/null || exit 1
fi

# =============================================================================
# Common paths
# =============================================================================
export LLAMA_PROJECT_ROOT="$PROJECT_ROOT"

# =============================================================================
# Helper: ensure ~/.drirc enables radv unified heap on APU
# =============================================================================
# RADV's default on APUs is a 2/3 DEVICE_LOCAL + 1/3 host split of the
# UMA heap. For llama.cpp this is almost always wrong: it causes the
# GPU allocator to place tensors in slow host memory (non-device-local)
# and can crash models > 2/3 of GPU-visible memory with DeviceLost.
# The radv_enable_unified_heap_on_apu driconf option merges the full
# VRAM+GTT pool into a single DEVICE_LOCAL heap.
_ensure_drirc_unified_heap() {
    local drirc="$HOME/.drirc"
    local need_create=0

    if [[ ! -f "$drirc" ]]; then
        need_create=1
    elif ! grep -q 'radv_enable_unified_heap_on_apu' "$drirc" 2>/dev/null; then
        need_create=1
    fi

    if [[ $need_create -eq 1 ]]; then
        cat > "$drirc" << 'DRICONF_EOF'
<?xml version="1.0" standalone="yes"?>
<!DOCTYPE driconf [
   <!ELEMENT driconf      (device+)>
   <!ELEMENT device       (application | engine)+>
   <!ATTLIST device       driver CDATA #IMPLIED
                          device CDATA #IMPLIED>
   <!ELEMENT application  (option+)>
   <!ATTLIST application  name CDATA #REQUIRED
                          executable CDATA #IMPLIED
                          executable_regexp CDATA #IMPLIED
                          sha1 CDATA #IMPLIED
                          application_name_match CDATA #IMPLIED
                          application_versions CDATA #IMPLIED>
   <!ELEMENT engine       (option+)>
   <!ATTLIST engine       engine_name_match CDATA #REQUIRED
                          engine_versions CDATA #IMPLIED>
   <!ELEMENT option       EMPTY>
   <!ATTLIST option       name CDATA #REQUIRED
                          value CDATA #REQUIRED>
]>
<driconf>
    <device driver="radv">
        <application name="llama-bench" application_name_match="llama-bench">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-server" application_name_match="llama-server">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-cli" application_name_match="llama-cli">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-batched-bench" application_name_match="llama-batched-bench">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
        <application name="llama-run" application_name_match="llama-run">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
    </device>
</driconf>
DRICONF_EOF
    fi
}

# =============================================================================
# Backend-specific setup
# =============================================================================
if [[ "$BACKEND" == "rocm" ]]; then
    export ROCM_PATH="$PROJECT_ROOT/deps"
    export HIP_PATH="$ROCM_PATH"
    export HIP_PLATFORM=amd
    export HSA_PATH="$ROCM_PATH"
    export ROCM_DIR="$ROCM_PATH"
    
    # Library paths
    export LD_LIBRARY_PATH="$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/llvm/lib/clang/23/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
    
    # Enable unified memory for APUs (uses GTT/system RAM via hipMallocManaged)
    export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
    
    # Binary paths (includes ROCm's bundled clang/lld)
    export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
    
    # Llama.cpp binary
    export LLAMA_BIN="$PROJECT_ROOT/src/llama-rocm/build/bin"
    export PATH="$LLAMA_BIN:$PATH"
    
    # gfx1151 (RDNA 3.5 / Strix Halo) has a HIP async-execution correctness bug:
    # batched inference can return garbage. HIP_LAUNCH_BLOCKING=1 serializes
    # kernel launches and restores correctness. This matches strix-llama.cpp's
    # ROCm CI configuration (see their README "ROCm and batched inference").
    # Per-token output is ~25-40% slower, but correctness is required.
    export HIP_LAUNCH_BLOCKING=1
    
    # Verify ROCm
    if [[ ! -d "$ROCM_PATH" ]]; then
        echo "ERROR: ROCm SDK not found at $ROCM_PATH"
        echo "Run ./scripts/rebuild.sh first"
        return 1 2>/dev/null || exit 1
    fi
    
    # Auto-detect GPU and set GFX version
    if [[ -f "$PROJECT_ROOT/scripts/detect-gpu.sh" ]]; then
        source "$PROJECT_ROOT/scripts/detect-gpu.sh"
    fi
    export HSA_OVERRIDE_GFX_VERSION="${LLAMA_GFX_VERSION:-11.0.3}"
    
    # Source ROCm environment if exists
    if [[ -f "$ROCM_PATH/etc/rocm.bashrc" ]]; then
        source "$ROCM_PATH/etc/rocm.bashrc" 2>/dev/null || true
    fi
    
    echo "ROCm environment loaded:"
    echo "  BACKEND=$BACKEND"
    echo "  ROCM_PATH=$ROCM_PATH"
    echo "  LLAMA_BIN=$LLAMA_BIN"
    echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    
elif [[ "$BACKEND" == "vulkan" ]]; then
    # Vulkan doesn't need ROCm paths
    export LLAMA_BIN="$PROJECT_ROOT/src/llama-vulkan/build/bin"
    export PATH="$LLAMA_BIN:$PATH"

    # Set Vulkan backend env var
    export GGML_BACKEND=vulkan

    # Ensure LD_LIBRARY_PATH is defined (required by llama-run.sh set -u)
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

    # Enable RADV unified heap on APU: makes the full VRAM+GTT pool
    # available as DEVICE_LOCAL memory. Without this, RADV splits the
    # UMA heap into 2/3 DEVICE_LOCAL + 1/3 host on Radeon 780M/890M
    # (e.g. 20 GiB DEVICE_LOCAL instead of 30 GiB), causing tensors to
    # spill into slow host memory and crashes for models > 2/3 of GPU-visible.
    # The ~/.drirc file is created if it doesn't exist; users can override
    # or extend it manually. For more details, see AGENTS.md "GPU detection".
    if [[ -z "${LLAMA_SKIP_UNIFIED_HEAP:-}" ]]; then
        _ensure_drirc_unified_heap
    fi

    echo "Vulkan environment loaded:"
    echo "  BACKEND=$BACKEND"
    echo "  LLAMA_BIN=$LLAMA_BIN"
    echo "  GGML_BACKEND=$GGML_BACKEND"

elif [[ "$BACKEND" == "metal" ]]; then
    # Metal uses macOS system frameworks; nothing to set besides LLAMA_BIN
    export LLAMA_BIN="$PROJECT_ROOT/src/llama-metal/build/bin"
    export PATH="$LLAMA_BIN:$PATH"

    # Metal uses unified memory automatically on Apple Silicon
    export GGML_METAL_DEVICE_DEBUG=0

    # Ensure LD_LIBRARY_PATH is defined (required by llama-run.sh set -u)
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

    echo "Metal environment loaded:"
    echo "  BACKEND=$BACKEND"
    echo "  LLAMA_BIN=$LLAMA_BIN"
    echo "  GGML_METAL_DEVICE_DEBUG=$GGML_METAL_DEVICE_DEBUG"
fi


# Alias for convenience
alias llama-server="$LLAMA_BIN/llama-server"
