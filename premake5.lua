-- Manifold2D standalone workspace -- deterministic 2D physics + geometry
-- (Box2D v3 model), zero external dependencies. Builds the library and its
-- Catch2 + rapidcheck test suite from the vendored deps; self-contained (the
-- vendored premake5.exe generates this). Static-CRT throughout -- there is no
-- DLL boundary here (that is the Arcane /MD consumer's concern, handled by
-- Arcane's own inline project, not this file).

workspace "Manifold2D"
    architecture "x64"
    configurations { "Debug", "Release", "Dist" }
    startproject "Manifold2DTests"
    location "."

    outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

    IncludeDir = {}
    IncludeDir["Manifold2D"]       = "include"
    IncludeDir["Mosaic"]           = "ThirdParty/Mosaic/include"
    IncludeDir["Catch2"]           = "vendor/Catch2/src"
    IncludeDir["rapidcheck"]       = "vendor/rapidcheck/include"
    IncludeDir["rapidcheck_catch"] = "vendor/rapidcheck/extras/catch/include"

    -- The vendored Catch2/rapidcheck wrappers honor these globals. Static CRT
    -- default (no DLL boundary); project files land in each dep's ide/ subdir.
    THIRDPARTY_STATICRUNTIME    = "on"
    THIRDPARTY_PROJECT_LOCATION = "ide"

    filter "system:windows"
        systemversion "latest"
    filter "system:linux"                    -- Part A SCAFFOLD (exercised in Part B)
        buildoptions { "-mavx2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra" }
        links { "pthread" }
    filter {}

    group "Dependencies"
        include "vendor/Catch2"
        include "vendor/rapidcheck"
    group ""

    project "Manifold2D"
        kind "StaticLib"
        language "C++"
        cppdialect "C++23"
        location "ide"
        staticruntime "on"
        floatingpoint "Strict"                -- determinism: /fp:fast is banned
        targetdir ("bin/" .. outputdir .. "/%{prj.name}")
        objdir    ("bin-int/" .. outputdir .. "/%{prj.name}")

        files { "include/**.hpp", "include/**.inl", "src/**.cpp" }
        includedirs { "%{IncludeDir.Manifold2D}", "%{IncludeDir.Mosaic}" }
        defines { "_CRT_SECURE_NO_WARNINGS" }

        filter "system:windows"
            buildoptions { "/Zc:__cplusplus", "/bigobj", "/arch:AVX2" }
        filter "configurations:Debug"
            runtime "Debug"   symbols "on"                  defines { "MANIFOLD2D_DEBUG" }
        filter "configurations:Release"
            runtime "Release" optimize "speed" symbols "on" defines { "MANIFOLD2D_RELEASE", "NDEBUG" }
        filter "configurations:Dist"
            runtime "Release" optimize "speed" symbols "off" defines { "MANIFOLD2D_DIST", "NDEBUG" }
        filter {}

    project "Manifold2DTests"
        kind "ConsoleApp"
        language "C++"
        cppdialect "C++23"
        location "ide"
        staticruntime "on"
        floatingpoint "Strict"
        targetdir ("bin/" .. outputdir .. "/%{prj.name}")
        objdir    ("bin-int/" .. outputdir .. "/%{prj.name}")

        files { "tests/**.hpp", "tests/**.inl", "tests/**.cpp" }
        includedirs {
            "%{IncludeDir.Manifold2D}",
            "%{IncludeDir.Mosaic}",
            "%{IncludeDir.Catch2}",
            "%{IncludeDir.rapidcheck}",
            "%{IncludeDir.rapidcheck_catch}",
            "tests",
        }
        links { "Manifold2D", "Catch2", "rapidcheck" }
        defines { "_CRT_SECURE_NO_WARNINGS" }

        filter "system:windows"
            buildoptions { "/Zc:__cplusplus", "/bigobj", "/arch:AVX2" }
        filter "configurations:Debug"
            runtime "Debug"   symbols "on"                  defines { "MANIFOLD2D_DEBUG" }
        filter "configurations:Release"
            runtime "Release" optimize "speed" symbols "on" defines { "MANIFOLD2D_RELEASE", "NDEBUG" }
        filter "configurations:Dist"
            runtime "Release" optimize "speed" symbols "off" defines { "MANIFOLD2D_DIST", "NDEBUG" }
        filter {}
