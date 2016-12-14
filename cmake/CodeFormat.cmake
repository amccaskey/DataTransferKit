if(NOT CLANG_FORMAT_EXECUTABLE)
    find_program(CLANG_FORMAT_EXECUTABLE
        NAMES
        clang-format-3.9
        clang-format-mp-3.9
    )
    if(CLANG_FORMAT_EXECUTABLE)
        message("-- Found clang-format: ${CLANG_FORMAT_EXECUTABLE}")
    else()
        message(FATAL_ERROR "-- clang-format not found")
    endif()
else()
    message("-- Using clang-format: ${CLANG_FORMAT_EXECUTABLE}")
    if(NOT EXISTS ${CLANG_FORMAT_EXECUTABLE})
        message(FATAL_ERROR "-- clang-format path is invalid")
    endif()
endif()

# Check that the vesion of clang-format is 3.9
execute_process(
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -version
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION
)
if(NOT CLANG_FORMAT_VERSION MATCHES "3.9")
    message(FATAL_ERROR "You must use clang-format version 3.9")
endif()
# Download diff-clang-format.py from ORNL-CEES/Cap
file(DOWNLOAD
    https://raw.githubusercontent.com/ORNL-CEES/Cap/master/diff-clang-format.py
    ${CMAKE_BINARY_DIR}/diff-clang-format.py
)
# Download docopt command line argument parser
file(DOWNLOAD
    https://raw.githubusercontent.com/docopt/docopt/0.6.2/docopt.py
    ${CMAKE_BINARY_DIR}/docopt.py
)
# Add a custom target that applies the C++ code formatting style to the source
add_custom_target(format-cpp
    ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/diff-clang-format.py
        --file-extension='.hpp'
        --file-extension='.cpp'
        --binary=${CLANG_FORMAT_EXECUTABLE}
        --style=file
        --config=${${PACKAGE_NAME}_SOURCE_DIR}/.clang-format
        --apply-patch
        ${${PACKAGE_NAME}_SOURCE_DIR}/packages
)
# Add a test that checks the code is formatted properly
file(WRITE
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_format_cpp.sh
    "#!/usr/bin/env bash\n"
    "\n"
    "${PYTHON_EXECUTABLE} "
    "${CMAKE_BINARY_DIR}/diff-clang-format.py "
    "--file-extension='.hpp' --file-extension='.cpp' "
    "--binary=${CLANG_FORMAT_EXECUTABLE} "
    "--style=file "
    "--config=${${PACKAGE_NAME}_SOURCE_DIR}/.clang-format "
    "${${PACKAGE_NAME}_SOURCE_DIR}/packages"
)
file(COPY
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_format_cpp.sh
    DESTINATION
        ${CMAKE_BINARY_DIR}
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)
add_test(
    NAME check_format_cpp
    COMMAND ${CMAKE_BINARY_DIR}/check_format_cpp.sh
)
