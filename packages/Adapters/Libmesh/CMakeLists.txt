##---------------------------------------------------------------------------##
## DataTransferKit/CMakeLists.txt
## Stuart R. Slattery
##---------------------------------------------------------------------------##
## TriBITS package file for the DataTransferKit package
##---------------------------------------------------------------------------##

##---------------------------------------------------------------------------##
## Define the package
##---------------------------------------------------------------------------##

TRIBITS_SUBPACKAGE(LibmeshAdapters)

##---------------------------------------------------------------------------##
## Add library, test, and examples.
##---------------------------------------------------------------------------##
include_directories(${CMAKE_SOURCE_DIR}/packages/Kokkos/src)

ADD_SUBDIRECTORY(src)

TRIBITS_ADD_TEST_DIRECTORIES(test)

IF(DTK_DATA_DIR)
    TRIBITS_ADD_EXAMPLE_DIRECTORIES(example)
ENDIF()

##---------------------------------------------------------------------------##
## D) Do standard postprocessing
##---------------------------------------------------------------------------##

TRIBITS_SUBPACKAGE_POSTPROCESS()

##---------------------------------------------------------------------------##
## end of DataTransferKit/CMakeLists.txt
##---------------------------------------------------------------------------##
