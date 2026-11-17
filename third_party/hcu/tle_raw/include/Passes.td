#ifndef HCU_TLE_RAW_PASSES
#define HCU_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def HCUMaterializeDeferredRaw
    : Pass<"hcu-tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw calls from compiled HIP sources";
  let description = [{
    Imports each compiled HIP source once and binds empty ``tle.dsl_region``
    calls carrying ``tle_raw.source_id`` to their specialized LLVM wrappers.
  }];
  let dependentDialects = ["mlir::triton::tle::TleDialect"];
}

#endif // HCU_TLE_RAW_PASSES
