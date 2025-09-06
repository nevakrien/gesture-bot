# ====== Config ======
TFLM_DIR := tflite-micro

# Auto-detect the MinGW-built TFLM static lib
TFLM_LIB := $(firstword \
  $(wildcard $(TFLM_DIR)/gen/*/lib/libtensorflow-microlite.a) \
  $(wildcard $(TFLM_DIR)/gen/**/libtensorflow-microlite.a) \
)
ifeq ($(strip $(TFLM_LIB)),)
$(error Could not find libtensorflow-microlite.a under $(TFLM_DIR)/gen/**/lib/. Build TFLM first)
endif

# Include paths for TFLM and its third-parties
TFLM_INC := \
  -I$(TFLM_DIR) \
  -I$(TFLM_DIR)/tensorflow \
  -I$(TFLM_DIR)/tensorflow/lite \
  -I$(TFLM_DIR)/tensorflow/lite/micro \
  -I$(TFLM_DIR)/third_party/flatbuffers/include \
  -I$(TFLM_DIR)/third_party/ruy \
  -I$(TFLM_DIR)/third_party/gemmlowp \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp

# OpenCV (MSYS2 pacman) — minimal, no pkg-config noise
PREFIX       := /mingw64
OPENCV_INC   := -I$(PREFIX)/include/opencv4
OPENCV_LIBDIR:= -L$(PREFIX)/lib
OPENCV_LIBS  := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

# ====== Build vars ======
CXX      := g++
CXXFLAGS := -O1 -std=c++17 $(TFLM_INC) $(OPENCV_INC)
LDFLAGS  := $(OPENCV_LIBDIR) $(OPENCV_LIBS) -L$(dir $(TFLM_LIB)) -ltensorflow-microlite -lws2_32

TARGET := cam_demo.exe
SRC    := src/cam_test.cpp

# ====== Rules ======
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)