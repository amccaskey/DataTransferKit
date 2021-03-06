INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# example executables
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  STKParallelSearch
  SOURCES parallel_search.cpp
  COMM mpi
  ARGS "--xml-in-file=input.xml"
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  STKInterpolation
  SOURCES interpolation.cpp
  COMM mpi
  ARGS "--xml-in-file=input.xml"
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  STKInlineInterpolation
  SOURCES inline_interpolation.cpp
  COMM mpi
  ARGS "--xml-in-file=input.xml"
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  STKRepeatedTransfer
  SOURCES repeated_transfer.cpp
  COMM mpi
  ARGS "--xml-in-file=input.xml"
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  STKMeshDeformation
  SOURCES mesh_deformation.cpp
  COMM mpi
  ARGS "--xml-in-file=input.xml"
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

# copy the input file for the search and interpolation examples
TRIBITS_COPY_FILES_TO_BINARY_DIR(
  STKInterpolationINP
  SOURCE_FILES input.xml
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  EXEDEPS STKInterpolation STKRepeatedTransfer STKInlineInterpolation STKParallelSearch
)

# copy the 2 mesh files for the search and interpolation examples
TRIBITS_COPY_FILES_TO_BINARY_DIR(
  STKInterpolationEXO
  SOURCE_FILES cube_mesh_small.exo pincell_mesh_small.exo
  SOURCE_DIR ${DTK_DATA_DIR}/exodus/
  DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  EXEDEPS STKInterpolation STKRepeatedTransfer STKInlineInterpolation STKParallelSearch
)

# copy the mesh for the mesh deformation example
TRIBITS_COPY_FILES_TO_BINARY_DIR(
  STKMeshDeformEXO
  SOURCE_FILES mesh_deform_2d.exo
  SOURCE_DIR ${DTK_DATA_DIR}/exodus/
  DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  EXEDEPS STKMeshDeformation
)
