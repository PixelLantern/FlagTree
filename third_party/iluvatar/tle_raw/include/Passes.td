#ifndef ILUVATAR_TLE_RAW_PASSES
#define ILUVATAR_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def IluvatarMaterializeDeferredRaw
    : Pass<"tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw calls from compiled Corex sources";
  let description = [{
    Imports each compiled Corex source once and constructs the bodies of
    ``iluvatar_tle.dsl_region`` calls carrying ``tle_raw.source_id`` using the
    imported entry function.
  }];
  let dependentDialects = ["mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

#endif // ILUVATAR_TLE_RAW_PASSES
