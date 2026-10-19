# Python Gil

The Global Interpreter Lock, or GIL, is a mutex in CPython that lets only one thread execute Python bytecode at a time. CPU-bound programs do not get faster with threads because of it. Use multiprocessing for CPU-bound work. Threads still help for I/O-bound tasks such as network calls, because the GIL is released while waiting.
