# === Standalone model downloader (no TF build, no Python) =====================
# Usage:
#   make            # downloads: person + hand_landmark_lite.tflite
#   make person     # just the person model
#   make hand       # just the hand-landmark (lite) model
#   make hand_full  # optional: full hand-landmark model (heavier)
#   make hand_task  # optional: MediaPipe .task bundle (for Tasks API)
#   make verify     # prints sha256 of downloaded files
#   make clean
#
# Works in MSYS/MinGW (Windows) or any POSIX shell with curl + sha256sum.

OUT_DIR ?= .
CURL    := curl -L --fail --retry 3 --retry-delay 1 --create-dirs -o

# -----------------------------------------------------------------------------#
# Person detection (Visual Wake Words, 96x96 INT8)
# Canonical small person/no-person model used in TFLite-Micro demos.
PERSON_TFLITE := $(OUT_DIR)/person_detect.tflite
PERSON_URL    := https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/models/person_detect.tflite

# -----------------------------------------------------------------------------#
# Hand models (MediaPipe)
# We fetch TFLite landmark models via jsDelivr CDN. You can pin a version:
#   make HANDS_VERSION=0.3            # or another published version
HANDS_VERSION ?=
HANDS_VER_SEG := $(if $(strip $(HANDS_VERSION)),@$(HANDS_VERSION),)

HAND_LITE_TFLITE := $(OUT_DIR)/hand_landmark_lite.tflite
HAND_LITE_URL    := https://cdn.jsdelivr.net/npm/@mediapipe/hands$(HANDS_VER_SEG)/hand_landmark_lite.tflite

HAND_FULL_TFLITE := $(OUT_DIR)/hand_landmark_full.tflite
HAND_FULL_URL    := https://cdn.jsdelivr.net/npm/@mediapipe/hands$(HANDS_VER_SEG)/hand_landmark_full.tflite

# # Optional: official MediaPipe Tasks bundle (.task) — use with MediaPipe Tasks C++ API
# HAND_TASK := $(OUT_DIR)/hand_landmarker.task
# HAND_TASK_URL := https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# HAND_TASK_SHA256 ?=  # (optional)

# -----------------------------------------------------------------------------#
.PHONY: all person hand hand_full hand_task verify clean
all: person hand

person: $(PERSON_TFLITE)
hand:   $(HAND_LITE_TFLITE)
hand_full: $(HAND_FULL_TFLITE)
# hand_task: $(HAND_TASK)

$(PERSON_TFLITE):
	@echo "↓ Download person model"
	@$(CURL) "$@" "$(PERSON_URL)"
ifneq ($(strip $(PERSON_SHA256)),)
	@echo "$(PERSON_SHA256) *$@" | sha256sum -c - || (rm -f "$@"; exit 1)
endif

$(HAND_LITE_TFLITE):
	@echo "↓ Download hand_landmark_lite.tflite (MediaPipe)"
	@$(CURL) "$@" "$(HAND_LITE_URL)"
ifneq ($(strip $(HAND_LITE_SHA256)),)
	@echo "$(HAND_LITE_SHA256) *$@" | sha256sum -c - || (rm -f "$@"; exit 1)
endif

$(HAND_FULL_TFLITE):
	@echo "↓ Download hand_landmark_full.tflite (MediaPipe)"
	@$(CURL) "$@" "$(HAND_FULL_URL)"
ifneq ($(strip $(HAND_FULL_SHA256)),)
	@echo "$(HAND_FULL_SHA256) *$@" | sha256sum -c - || (rm -f "$@"; exit 1)
endif

# $(HAND_TASK):
# 	@echo "↓ Download MediaPipe Hand Landmarker (.task)"
# 	@$(CURL) "$@" "$(HAND_TASK_URL)"
# ifneq ($(strip $(HAND_TASK_SHA256)),)
# 	@echo "$(HAND_TASK_SHA256) *$@" | sha256sum -c - || (rm -f "$@"; exit 1)
# endif

clean:
	@rm -f "$(PERSON_TFLITE)" "$(HAND_LITE_TFLITE)" "$(HAND_FULL_TFLITE)" #"$(HAND_TASK)"
