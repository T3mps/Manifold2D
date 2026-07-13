
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Hand-rolled materialized config for premake-built Catch2.
 *
 * Upstream generates this file from `catch_user_config.hpp.in` at
 * CMake-configure time, substituting `#cmakedefine` lines based on the
 * CMake cache. Since we build Catch2 with premake5 (not CMake), the
 * `.in` template never gets rendered. This file is the rendered default:
 * no opt-in/opt-out feature macros defined, default reporter "console",
 * default console width 80. Override by defining the desired macros in
 * the premake project before any Catch2 header is included.
 */

#ifndef CATCH_USER_CONFIG_HPP_INCLUDED
#define CATCH_USER_CONFIG_HPP_INCLUDED


// ------
// Overridable compilation flags — left undefined to take Catch2's
// own platform-detected defaults. Define them in the premake project
// (via `defines { ... }`) to force a feature on or off.
// ------

// CATCH_CONFIG_ANDROID_LOGWRITE / CATCH_CONFIG_NO_ANDROID_LOGWRITE
// CATCH_CONFIG_COLOUR_WIN32 / CATCH_CONFIG_NO_COLOUR_WIN32
// CATCH_CONFIG_COUNTER / CATCH_CONFIG_NO_COUNTER
// CATCH_CONFIG_CPP11_TO_STRING / CATCH_CONFIG_NO_CPP11_TO_STRING
// CATCH_CONFIG_CPP17_BYTE / CATCH_CONFIG_NO_CPP17_BYTE
// CATCH_CONFIG_CPP17_OPTIONAL / CATCH_CONFIG_NO_CPP17_OPTIONAL
// CATCH_CONFIG_CPP17_STRING_VIEW / CATCH_CONFIG_NO_CPP17_STRING_VIEW
// CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS / CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
// CATCH_CONFIG_CPP17_VARIANT / CATCH_CONFIG_NO_CPP17_VARIANT
// CATCH_CONFIG_GLOBAL_NEXTAFTER / CATCH_CONFIG_NO_GLOBAL_NEXTAFTER
// CATCH_CONFIG_POSIX_SIGNALS / CATCH_CONFIG_NO_POSIX_SIGNALS
// CATCH_CONFIG_GETENV / CATCH_CONFIG_NO_GETENV
// CATCH_CONFIG_USE_ASYNC / CATCH_CONFIG_NO_USE_ASYNC
// CATCH_CONFIG_WCHAR / CATCH_CONFIG_NO_WCHAR
// CATCH_CONFIG_WINDOWS_SEH / CATCH_CONFIG_NO_WINDOWS_SEH
// CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT / CATCH_CONFIG_NO_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT
// CATCH_CONFIG_USE_BUILTIN_CONSTANT_P / CATCH_CONFIG_NO_USE_BUILTIN_CONSTANT_P
// CATCH_CONFIG_DEPRECATION_ANNOTATIONS / CATCH_CONFIG_NO_DEPRECATION_ANNOTATIONS
// CATCH_CONFIG_THREAD_SAFE_ASSERTIONS / CATCH_CONFIG_NO_THREAD_SAFE_ASSERTIONS


// ------
// Simple toggle defines — all opt-in, left undefined here.
// ------

// CATCH_CONFIG_BAZEL_SUPPORT
// CATCH_CONFIG_DISABLE_EXCEPTIONS
// CATCH_CONFIG_DISABLE_EXCEPTIONS_CUSTOM_HANDLER
// CATCH_CONFIG_DISABLE
// CATCH_CONFIG_DISABLE_STRINGIFICATION
// CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS
// CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER
// CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER
// CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER
// CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER
// CATCH_CONFIG_EXPERIMENTAL_REDIRECT
// CATCH_CONFIG_FAST_COMPILE
// CATCH_CONFIG_NOSTDOUT
// CATCH_CONFIG_PREFIX_ALL
// CATCH_CONFIG_PREFIX_MESSAGES
// CATCH_CONFIG_WINDOWS_CRTDBG
// CATCH_CONFIG_SHARED_LIBRARY


// ------
// "Variable" defines — these must have values.
// ------

#define CATCH_CONFIG_DEFAULT_REPORTER "console"
#define CATCH_CONFIG_CONSOLE_WIDTH 80

// CATCH_CONFIG_FALLBACK_STRINGIFIER — intentionally not defined; Catch2
// expects users to set this only if they want it. Define it in the
// premake project if needed.

#endif // CATCH_USER_CONFIG_HPP_INCLUDED
