
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  BoxGeometry_test
  SOURCES tstBoxGeometry.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  CylinderGeometry_test
  SOURCES tstCylinderGeometry.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Point_test
  SOURCES tstPoint.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  BasicEntitySet_test
  SOURCES tstBasicEntitySet.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  BasicGeometryLocalMap_test
  SOURCES tstBasicGeometryLocalMap.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  EntityCenteredShapeFunction_test
  SOURCES tstEntityCenteredShapeFunction.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  EntityCenteredField_test
  SOURCES tstEntityCenteredField.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

