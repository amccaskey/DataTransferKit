IF(MOAB_LIBRARY_DIRS)
  SET(MOAB_LIBRARY_DIR ${MOAB_LIBRARY_DIRS})
  SET(MOAB_DIR ${MOAB_LIBRARY_DIRS})
ENDIF()

IF(TPL_MOAB_LIBRARY_DIRS)
  SET(MOAB_LIBRARY_DIR ${TPL_MOAB_LIBRARY_DIRS})
  SET(MOAB_DIR ${TPL_MOAB_LIBRARY_DIRS})
ENDIF()

IF(MOAB_INCLUDE_DIRS)
  SET(MOAB_INCLUDE_DIR ${MOAB_INCLUDE_DIRS})
ENDIF()

IF(TPL_MOAB_INCLUDE_DIRS)
  SET(MOAB_INCLUDE_DIR ${TPL_MOAB_INCLUDE_DIRS})
ENDIF()

IF(TPL_MOAB_LIBRARIES)
  SET(MOAB_LIBRARIES ${TPL_MOAB_LIBRARIES})
ENDIF()

FIND_PACKAGE(MOAB)

IF(MOAB_FOUND)

  IF(NOT(MOAB_INCLUDE_DIRS))
    SET(MOAB_INCLUDE_DIRS ${MOAB_INCLUDES})
  ENDIF()

  IF(NOT(MOAB_LIBRARY_DIRS))
    SET(MOAB_LIBRARY_DIRS ${MOAB_LIBRARY_DIR})
  ENDIF()

  IF(NOT(TPL_MOAB_INCLUDE_DIRS))
    ADVANCED_SET(TPL_MOAB_INCLUDE_DIRS ${MOAB_INCLUDE_DIRS} CACHE PATH
      "Set inside of FindTPLMOAB.cmake")
  ENDIF()

  IF(NOT(TPL_MOAB_LIBRARY_DIRS))
    SET(TPL_MOAB_LIBRARY_DIRS ${MOAB_LIBRARY_DIRS})
  ENDIF()

ENDIF()

TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES( 
  MOAB
  REQUIRED_HEADERS MBInterface.hpp MBParallelComm.hpp
  REQUIRED_LIBS_NAMES MOAB
  )
