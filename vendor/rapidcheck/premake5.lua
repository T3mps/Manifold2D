-- rapidcheck premake5 build script
-- BSD-2-Clause property-based testing library for C++
--
-- Included by BOTH the Server (Aphelyon) and Arcane workspaces. The two
-- globals below default to the historical Server behavior; the Arcane
-- workspace overrides them (dynamic CRT, project files in ide-md/) so
-- the workspaces never overwrite each other's generated vcxproj and
-- never collide on lib outputs (outputdir differs per workspace).

project "rapidcheck"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    location(THIRDPARTY_PROJECT_LOCATION or ".")
    staticruntime(THIRDPARTY_STATICRUNTIME or "on")

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    files {
        "src/**.cpp",
        "include/**.h",
        "include/**.hpp",
    }

    includedirs { "include" }

    defines { "_CRT_SECURE_NO_WARNINGS" }

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
