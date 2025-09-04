# ===== User options =====
PYTHON ?= python3
CXX    ?= c++
SOURCES := machine_alpha.cpp $(wildcard cpp_features/*.cpp)

# ===== Auto-detected Python/pybind11 flags =====
EXT        := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || \
                 $(PYTHON) -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')
PYINCLUDE  := $(shell $(PYTHON) -m pybind11 --includes)
PYCFLAGS   := $(shell $(PYTHON)-config --cflags)
PYLDFLAGS  := $(shell $(PYTHON)-config --ldflags)

# ===== Build flags =====
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?=

# ===== Optional OpenMP =====
UNAME_S := $(shell uname -s)
ifeq ($(USE_OMP),1)
  ifeq ($(UNAME_S),Darwin)
    # macOS: brew install libomp
    OMP_CXXFLAGS += -Xpreprocessor -fopenmp
    OMP_LDFLAGS  += -lomp
  else
    # Linux
    OMP_CXXFLAGS += -fopenmp
    OMP_LDFLAGS  += -fopenmp
  endif
endif

# ===== Targets =====
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := machine_alpha$(EXT)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(OMP_LDFLAGS) $(PYLDFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(OMP_CXXFLAGS) $(PYINCLUDE) $(PYCFLAGS)

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
