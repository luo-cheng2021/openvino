# Profiler based on Chrome Tracing
Chrome tracing is a profiling tool which is easily accessible at URL `chrome://tracing/` in Chrome browser. CPU plugin compiled with `-DENABLE_CPU_PROFILER=ON` has the capability of generating such json-format tracing logs that can be loaded into & viewed with this powerful tool. 

It cannot replace ITT & Vtune based profiling, but it provides a very convenient & customizable alternative.

set `OV_CPU_PROFILE` environment variable to `1` will enable the profiling and tracing log generation.