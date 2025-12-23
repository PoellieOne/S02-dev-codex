# GEMINI.md: Project Overview and Development Guide

## Project Overview

This project, named **SoRa S02 · Symphonia**, is a Python-based system for real-time processing of sensor data from a custom hardware device called **Core-0**. The system communicates with Core-0 over a UART serial connection, parsing a custom binary protocol to receive sensor events.

The core of the project is a multi-stage processing pipeline designed to analyze rotational movement. The pipeline takes raw sensor events and performs the following steps:

1.  **Cycle Detection:** Identifies fundamental 3-point cycles from the stream of sensor events.
2.  **Tile Aggregation:** Groups cycles into time-based "tiles".
3.  **Compass Direction:** Determines the rotational direction (Clockwise, Counter-Clockwise, or Undecided) from the aggregated tile data.
4.  **Movement State:** Tracks the overall movement state, including RPM, total rotations, and whether the system is `STILL` or in `MOVEMENT`.

The project includes a real-time command-line interface (CLI) to visualize the state of the pipeline and can log detailed data to JSONL files for offline analysis. The system is configurable through different profiles (e.g., `production`, `bench`) to tune the pipeline's parameters for various scenarios.

### Key Files

*   `sym_cycles/realtime_states_v1_9_canonical.py`: The heart of the processing pipeline, defining the `RealtimePipeline` and its constituent components (`CyclesState`, `TilesState`, `InertialCompass`, `MovementBody`).
*   `scripts/live_symphonia_v2_0.py`: The main application for connecting to the Core-0 device, running the pipeline, and displaying the real-time UI.
*   `sym_cycles/l1_physical_activity.py`: Implements a higher-level "L1" state machine for physical activity detection.
*   `Task Brief v1.0.md`: A template for defining development tasks, indicating a structured development process with clear roles for AI agents and human verifiers.

## Building and Running

This is a Python project. To run the main application, you will need to have Python 3 and `pyserial` installed.

### Dependencies

The primary dependency is `pyserial`. You can likely install it with pip:

```bash
pip install pyserial
```

### Running the Live Application

The main entry point for running the live processing application is `scripts/live_symphonia_v2_0.py`. You will need to provide the correct serial port for your Core-0 device.

```bash
# Example of running the live application
python3 scripts/live_symphonia_v2_0.py --port /dev/ttyUSB0
```

You can use the `--help` flag to see all available options:

```bash
python3 scripts/live_symphonia_v2_0.py --help
```

Common options include:

*   `--port`: The serial port to connect to.
*   `--baud`: The baud rate for the serial connection.
*   `--profile`: The pipeline profile to use (`production`, `bench`, `bench_tolerant`).
*   `--log`: Enable logging of the pipeline state to a `.jsonl` file.

## Development Conventions

The codebase demonstrates a set of strong development conventions:

*   **Structured and Modular:** The code is well-organized into modules with clear responsibilities (e.g., `CyclesState`, `TilesState`).
*   **Clear Naming:** File and class names are descriptive and versioned (e.g., `realtime_states_v1_9_canonical.py`), indicating an iterative development process.
*   **Robustness:** The use of a canonicalization function (`canon_event24`) to standardize input data shows a focus on creating a robust and resilient pipeline.
*   **Debugging and Telemetry:** The "TruthProbe" feature provides detailed telemetry on why data is rejected, which is invaluable for debugging and fine-tuning the pipeline.
*   **Task-Based Workflow:** The `Task Brief v1.0.md` file suggests a highly structured, formal development process where tasks are clearly defined, scoped, and have explicit acceptance criteria. This process seems to be designed for collaboration between human developers and AI agents.
*   **No Explicit Tests:** While the project has a very structured approach, there are no obvious unit tests (e.g., in a `tests/` directory or using a framework like `pytest`). Verification seems to be done through live testing, analysis of logged data, and the structured task-based workflow.

For future development, it is recommended to adhere to these existing conventions.
