add_triton_library(HCURawWrapperAdapter
  RawWrapperAdapter.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRDataLayoutInterfaces
)

target_include_directories(HCURawWrapperAdapter PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party
)

add_triton_library(HCUTLERawTransforms
  MaterializeDeferredRaw.cpp

  DEPENDS
  HCUTLERawTransformsIncGen
  TleTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  TritonTLERawUtils
  HCURawWrapperAdapter
)

target_include_directories(HCUTLERawTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/hcu/tle_raw/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
  ${CMAKE_SOURCE_DIR}/third_party
)
