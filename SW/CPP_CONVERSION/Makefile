# Makefile for Raspberry Pi 4/5 ARM Cortex-A72 Real-Time Diagnosis Engine
CXX = g++
CXXFLAGS = -O3 -std=c++17 -pthread -Wall -Wextra -march=native
TARGET = gearbox_realtime_diagnosis_engine
SRC = gearbox_realtime_diagnosis_engine.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "========================================================="
	@echo " [SUCCESS] Built $(TARGET) for ARM Cortex-A72 / RPi 4/5"
	@echo "========================================================="

clean:
	rm -f $(TARGET) *.o *.so
