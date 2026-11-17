#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "nvidia/tle_raw/include/CompiledRaw.h"
#include "nvidia/tle_raw/include/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/TleRawMaterialize.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <memory>

namespace tle = mlir::triton::tle;
namespace nv_tle_raw = mlir::triton::nvidia::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_NVIDIAMATERIALIZEDEFERREDRAW
#include "nvidia/tle_raw/include/Passes.h.inc"

class NvidiaMaterializeDeferredRawPass
    : public impl::NvidiaMaterializeDeferredRawBase<
          NvidiaMaterializeDeferredRawPass> {
public:
  using impl::NvidiaMaterializeDeferredRawBase<
      NvidiaMaterializeDeferredRawPass>::NvidiaMaterializeDeferredRawBase;

  NvidiaMaterializeDeferredRawPass() = default;
  explicit NvidiaMaterializeDeferredRawPass(
      llvm::StringMap<nv_tle_raw::CompiledRaw> sources)
      : sources(
            std::make_shared<const llvm::StringMap<nv_tle_raw::CompiledRaw>>(
                std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const auto &compiled = *sources;

    // Import each source once. These maps live only for this pass execution.
    llvm::StringMap<llvm::StringMap<Operation *>> imported;
    for (const auto &[sourceId, entry] : compiled) {
      if (entry.llvmIr.empty() || entry.dslFileName.empty()) {
        module.emitError("incomplete compiled raw source ") << sourceId;
        return signalPassFailure();
      }
      auto parsed =
          tle::raw::parseLLVMModule(module.getContext(), entry.llvmIr);
      if (!parsed) {
        module.emitError("cannot parse compiled raw source ") << sourceId;
        return signalPassFailure();
      }

      // Fill only directly unknown locations, once per compiled source.
      auto fallbackLoc = FileLineColRange::get(
          StringAttr::get(module.getContext(), entry.dslFileName));
      parsed->walk([&](Operation *op) {
        if (isa<UnknownLoc>(op->getLoc()))
          op->setLoc(fallbackLoc);
      });

      llvm::SHA256 hasher;
      hasher.update(entry.dslFileName);
      std::string globalPrefix =
          "__tle_raw_global_" + llvm::toHex(hasher.final()) + "_";
      std::string functionPrefix = "__tle_raw_fn_" + sourceId.str() + "_";
      auto renamed = tle::raw::renameLLVMSymbols(
          *parsed, [&](Operation *symbol) -> StringAttr {
            StringRef name = SymbolTable::getSymbolName(symbol).getValue();
            if (auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol)) {
              if (function.isExternal() ||
                  name.starts_with("__tle_raw_wrapper_"))
                return StringAttr::get(module.getContext(), name);
              return StringAttr::get(module.getContext(),
                                     functionPrefix + name.str());
            }
            auto global = cast<LLVM::GlobalOp>(symbol);
            if (!global.getValueOrNull() &&
                global.getInitializerRegion().empty())
              return StringAttr::get(module.getContext(), name);
            return StringAttr::get(module.getContext(),
                                   globalPrefix + name.str());
          });
      if (failed(renamed)) {
        module.emitError("cannot rename compiled raw source ") << sourceId;
        return signalPassFailure();
      }

      // Raw function definitions belong to this kernel. Inlining may make
      // originals dead; private visibility lets symbol DCE remove them.
      for (auto function : parsed->getOps<LLVM::LLVMFuncOp>()) {
        if (!function.isExternal()) {
          function.setLinkage(LLVM::Linkage::Internal);
          SymbolTable::setSymbolVisibility(function,
                                           SymbolTable::Visibility::Private);
        }
      }

      auto cloned = tle::raw::cloneLLVMSymbols(module, *parsed);
      if (failed(cloned)) {
        module.emitError("cannot import compiled raw source ") << sourceId;
        return signalPassFailure();
      }

      auto &symbols = imported[sourceId];
      for (const auto &item : *renamed)
        symbols[item.getKey()] = cloned->lookup(item.getValue());
    }

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result = module.walk([&](tle::DSLRegionOp op) -> WalkResult {
      auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
      if (!sourceIdAttr || !op.getBody().empty())
        return WalkResult::advance();
      StringRef sourceId = sourceIdAttr.getValue();
      auto source = imported.find(sourceId);
      if (source == imported.end()) {
        op.emitError("missing imported raw source ") << sourceId;
        return WalkResult::interrupt();
      }
      auto wrapper =
          op->getAttrOfType<FlatSymbolRefAttr>("tle_raw.wrapper_symbol");
      if (!wrapper) {
        op.emitError("missing raw wrapper symbol for source ") << sourceId;
        return WalkResult::interrupt();
      }
      auto function = dyn_cast_or_null<LLVM::LLVMFuncOp>(
          source->second.lookup(wrapper.getValue()));
      if (!function || function.isExternal()) {
        op.emitError("cannot find raw wrapper definition ")
            << wrapper.getValue() << " in source " << sourceId;
        return WalkResult::interrupt();
      }
      op->setAttr(
          "tle_raw.wrapper_symbol",
          FlatSymbolRefAttr::get(module.getContext(), function.getSymName()));
      op->removeAttr(kSourceIdAttr);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  std::shared_ptr<const llvm::StringMap<nv_tle_raw::CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<nv_tle_raw::CompiledRaw>>();
};

std::unique_ptr<Pass> createNvidiaMaterializeDeferredRaw(
    llvm::StringMap<nv_tle_raw::CompiledRaw> sources) {
  return std::make_unique<NvidiaMaterializeDeferredRawPass>(std::move(sources));
}

} // namespace mlir
