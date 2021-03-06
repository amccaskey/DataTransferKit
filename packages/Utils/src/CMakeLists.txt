
TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  DTK_DBC.hpp
  DTK_PredicateComposition.hpp
  DTK_PredicateComposition_impl.hpp
  DTK_SanitizerMacros.hpp
  DTK_SearchTreeFactory.hpp
  DTK_StaticSearchTree.hpp
  DTK_StaticSearchTree_impl.hpp
  )

APPEND_SET(SOURCES
  DTK_DBC.cpp
  DTK_SearchTreeFactory.cpp
  )

SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR}/Nanoflann)
APPEND_SET(HEADERS
  ${DIR}/DTK_nanoflann.hpp
  )

TRIBITS_ADD_LIBRARY(
  dtk_utils
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )
