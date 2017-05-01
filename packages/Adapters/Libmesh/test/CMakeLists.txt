
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Adjacencies_test
  SOURCES tstAdjacencies.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshEntity_test
  SOURCES tstLibmeshEntity.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshEntityIterator_test
  SOURCES tstLibmeshEntityIterator.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshEntitySet_test
  SOURCES tstLibmeshEntitySet.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshEntityLocalMap_test
  SOURCES tstLibmeshEntityLocalMap.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshNodalShapeFunction_test
  SOURCES tstLibmeshNodalShapeFunction.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshVariableField_test
  SOURCES tstLibmeshVariableField.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshVariableField2_test
  SOURCES tstLibmeshVariableField2.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LibmeshEntityIntegrationRule_test
  SOURCES tstLibmeshEntityIntegrationRule.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
include(boost_helper)
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Kokkos_LIBRARY_DIRS})
link_directories(${DataTransferKit_LIBRARY_DIRS})
add_boost_tests(tstLibmeshUserApplication.cpp "${CMAKE_SOURCE_DIR}/packages/Kokkos/src;${TPL_Libmesh_INCLUDE_DIRS};${DataTransferKitKokkos_INCLUDE_DIRS};${Kokkos_INCLUDE_DIRS}" "${Boost_LIBRARIES};${TPL_Libmesh_LIBRARIES};${Kokkos_LIBRARIES};${Teuchos_LIBRARIES};${DataTransferKit_LIBRARIES}")
