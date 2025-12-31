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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "pypto/core/error.h"

namespace py = pybind11;

// Helper functions to demonstrate error raising from C++
void raise_value_error(const std::string& message) { throw pypto::ValueError(message); }

void raise_type_error(const std::string& message) { throw pypto::TypeError(message); }

void raise_runtime_error(const std::string& message) { throw pypto::RuntimeError(message); }

void raise_not_implemented_error(const std::string& message) { throw pypto::NotImplementedError(message); }

void raise_index_error(const std::string& message) { throw pypto::IndexError(message); }

void raise_generic_error(const std::string& message) { throw pypto::Error(message); }

int divide(int a, int b) {
  if (b == 0) {
    throw pypto::ValueError("Division by zero is not allowed");
  }
  return a / b;
}

int get_array_element(int index) {
  const int size = 5;
  if (index > 0) {
    return get_array_element(index - 1);
  } else {
    raise_index_error("Hello error message");
  }
}

PYBIND11_MODULE(_pypto_core, m) {
  m.doc() = "PyPTO core module with error handling";

  // Register custom exception types and map them to Python exceptions
  static py::exception<pypto::Error> exc_error(m, "Error", PyExc_Exception);
  static py::exception<pypto::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
  static py::exception<pypto::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
  static py::exception<pypto::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
  static py::exception<pypto::NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                             PyExc_NotImplementedError);
  static py::exception<pypto::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const pypto::ValueError& e) {
      PyErr_SetString(PyExc_ValueError, e.GetFullMessage().c_str());
    } catch (const pypto::TypeError& e) {
      PyErr_SetString(PyExc_TypeError, e.GetFullMessage().c_str());
    } catch (const pypto::RuntimeError& e) {
      PyErr_SetString(PyExc_RuntimeError, e.GetFullMessage().c_str());
    } catch (const pypto::NotImplementedError& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.GetFullMessage().c_str());
    } catch (const pypto::IndexError& e) {
      PyErr_SetString(PyExc_IndexError, e.GetFullMessage().c_str());
    } catch (const pypto::Error& e) {
      PyErr_SetString(PyExc_Exception, e.GetFullMessage().c_str());
    }
  });

  // Expose helper functions for testing error handling
  m.def("raise_value_error", &raise_value_error, "Raise a ValueError with the given message",
        py::arg("message"));

  m.def("raise_type_error", &raise_type_error, "Raise a TypeError with the given message",
        py::arg("message"));

  m.def("raise_runtime_error", &raise_runtime_error, "Raise a RuntimeError with the given message",
        py::arg("message"));

  m.def("raise_not_implemented_error", &raise_not_implemented_error,
        "Raise a NotImplementedError with the given message", py::arg("message"));

  m.def("raise_index_error", &raise_index_error, "Raise an IndexError with the given message",
        py::arg("message"));

  m.def("raise_generic_error", &raise_generic_error, "Raise a generic Error with the given message",
        py::arg("message"));

  m.def("divide", &divide, "Divide two integers, raises ValueError if divisor is zero", py::arg("a"),
        py::arg("b"));

  m.def("get_array_element", &get_array_element,
        "Get element from a fixed array, raises IndexError if index is out of "
        "bounds",
        py::arg("index"));

  // Expose StackFrame class
  py::class_<pypto::StackFrame>(m, "StackFrame")
      .def_readonly("function", &pypto::StackFrame::function)
      .def_readonly("filename", &pypto::StackFrame::filename)
      .def_readonly("lineno", &pypto::StackFrame::lineno)
      .def_readonly("pc", &pypto::StackFrame::pc)
      .def("to_string", &pypto::StackFrame::to_string)
      .def("__str__", &pypto::StackFrame::to_string)
      .def("__repr__", &pypto::StackFrame::to_string);

  // Helper function to get stack trace from a C++ exception
  m.def(
      "get_cpp_stack_trace",
      []() -> std::vector<pypto::StackFrame> { return pypto::Backtrace::GetInstance().CaptureStackTrace(1); },
      "Capture and return the current C++ stack trace");
}
