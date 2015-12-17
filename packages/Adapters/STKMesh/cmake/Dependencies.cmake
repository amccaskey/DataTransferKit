TRIBITS_PACKAGE_DEFINE_DEPENDENCIES(
  LIB_REQUIRED_PACKAGES
    DataTransferKitUtils
    DataTransferKitInterface
    DataTransferKitIntrepidAdapters
    Teuchos 
    Tpetra
    Shards
    Intrepid
    STKUtil
    STKTopology
    STKMesh
  TEST_REQUIRED_PACKAGES
    DataTransferKitOperators
    STKIO
  EXAMPLE_REQUIRED_PACKAGES
    STKUnit_tests
  LIB_REQUIRED_TPLS
    MPI
  )
