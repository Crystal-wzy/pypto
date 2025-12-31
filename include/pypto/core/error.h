/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_CORE_ERROR_H_
#define PYPTO_CORE_ERROR_H_

#include <backtrace.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pypto {

// Stack frame information
struct StackFrame {
  std::string function;
  std::string filename;
  int lineno;
  uintptr_t pc;

  StackFrame() : lineno(0), pc(0) {}
  StackFrame(std::string func, std::string file, int line, uintptr_t program_counter)
      : function(std::move(func)), filename(std::move(file)), lineno(line), pc(program_counter) {}

  std::string to_string() const;
};

// Backtrace capture and formatting
class Backtrace {
 public:
  static Backtrace& GetInstance();

  // Capture the current stack trace, skipping 'skip' frames
  std::vector<StackFrame> CaptureStackTrace(int skip = 0);

  // Format stack frames as a string
  static std::string FormatStackTrace(const std::vector<StackFrame>& frames);

 public:
  Backtrace();
  ~Backtrace() = default;
  Backtrace(const Backtrace&) = delete;
  Backtrace& operator=(const Backtrace&) = delete;

 private:
  backtrace_state* state_;

  static void ErrorCallback(void* data, const char* msg, int errnum);
  static int FullCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function);
};

// Base error class with stack trace support
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message, int skip_frames = 1);
  explicit Error(const char* message, int skip_frames = 1);

  const std::vector<StackFrame>& GetStackTrace() const { return stack_trace_; }
  std::string GetFormattedStackTrace() const;
  std::string GetFullMessage() const;

 private:
  std::vector<StackFrame> stack_trace_;
};

// Specific error types
class ValueError : public Error {
 public:
  explicit ValueError(const std::string& message) : Error(message, 2) {}
  explicit ValueError(const char* message) : Error(message, 2) {}
};

class TypeError : public Error {
 public:
  explicit TypeError(const std::string& message) : Error(message, 2) {}
  explicit TypeError(const char* message) : Error(message, 2) {}
};

class RuntimeError : public Error {
 public:
  explicit RuntimeError(const std::string& message) : Error(message, 2) {}
  explicit RuntimeError(const char* message) : Error(message, 2) {}
};

class NotImplementedError : public Error {
 public:
  explicit NotImplementedError(const std::string& message) : Error(message, 2) {}
  explicit NotImplementedError(const char* message) : Error(message, 2) {}
};

class IndexError : public Error {
 public:
  explicit IndexError(const std::string& message) : Error(message, 2) {}
  explicit IndexError(const char* message) : Error(message, 2) {}
};

}  // namespace pypto

#endif  // PYPTO_CORE_ERROR_H_
