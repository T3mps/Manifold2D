-- Catch2 v3 premake5 build script
-- Boost Software License 1.0 -- modern C++ unit-testing framework
--
-- Included by BOTH the Server (Aphelyon) and Arcane workspaces. The two
-- globals below default to the historical Server behavior; the Arcane
-- workspace overrides them (dynamic CRT, project files in ide-md/) so
-- the workspaces never overwrite each other's generated vcxproj and
-- never collide on lib outputs (outputdir differs per workspace).

project "Catch2"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    location(THIRDPARTY_PROJECT_LOCATION or ".")
    staticruntime(THIRDPARTY_STATICRUNTIME or "on")

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir    ("bin-int/" .. outputdir .. "/%{prj.name}")

    files {
        "src/catch2/**.cpp",
        "src/catch2/**.hpp",
    }

    -- catch_main.cpp provides a default main() that conflicts with
    -- test executables that define their own; exclude it here.
    -- Tests that want the default runner can link Catch2WithDefs separately
    -- (a future task); for now all test executables provide their own main.
    removefiles { "src/catch2/internal/catch_main.cpp" }

    includedirs { "src" }

    defines { "_CRT_SECURE_NO_WARNINGS", "NOMINMAX" }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
        defines { "NDEBUG" }

    filter "configurations:Dist"
        runtime "Release"
        optimize "on"
        defines { "NDEBUG" }
