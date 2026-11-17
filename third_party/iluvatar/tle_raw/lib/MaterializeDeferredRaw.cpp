#include "iluvatar/tle/dialect/include/IR/Dialect.h"
#include "iluvatar/tle/utils/include/TleRawMaterialize.h"
#include "iluvatar/tle_raw/include/CompiledRaw.h"
#include "iluvatar/tle_raw/include/Passes.h"
#include "ir.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <memory>

namespace iluvatar_tle = mlir::triton::iluvatar_tle;
using mlir::triton::iluvatar::tle_raw::CompiledRaw;

namespace mlir {

#define GEN_PASS_DEF_ILUVATARMATERIALIZEDEFERREDRAW
#include "iluvatar/tle_raw/include/Passes.h.inc"

class IluvatarMaterializeDeferredRawPass
    : public impl::IluvatarMaterializeDeferredRawBase<
          IluvatarMaterializeDeferredRawPass> {
public:
  using impl::IluvatarMaterializeDeferredRawBase<
      IluvatarMaterializeDeferredRawPass>::IluvatarMaterializeDeferredRawBase;

  IluvatarMaterializeDeferredRawPass() = default;

  explicit IluvatarMaterializeDeferredRawPass(
      llvm::StringMap<CompiledRaw> sources)
      : sources(std::make_shared<const llvm::StringMap<CompiledRaw>>(
            std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const auto &compiled = *sources;

    // Import each source once. These maps live only for this pass execution.
    llvm::StringMap<llvm::StringMap<Operation *>> imported;
    for (const auto &[sourceId, entry] : compiled) {
      if (entry.llvmIr.empty() || entry.dslFileName.empty() ||
          entry.externFuncName.empty()) {
        module.emitError("incomplete compiled raw source ") << sourceId;
        return signalPassFailure();
      }
      auto parsed =
          iluvatar_tle::raw::parseLLVMModule(module.getContext(), entry.llvmIr);
      if (!parsed) {
        module.emitError("cannot parse compiled raw source ") << sourceId;
        return signalPassFailure();
      }
      // Resolve the source-level entry name before symbols receive prefixes.
      auto entryFunction = iluvatar_tle::raw::findExternalLLVMFunc(
          *parsed, StringRef(entry.externFuncName));
      if (!entryFunction || entryFunction.isExternal()) {
        module.emitError("cannot find raw entry definition ")
            << entry.externFuncName << " in source " << sourceId;
        return signalPassFailure();
      }
      std::string entrySymbol = entryFunction.getSymName().str();
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
      auto renamed = iluvatar_tle::raw::renameLLVMSymbols(
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
          function.setVisibility_(LLVM::Visibility::Default);
          SymbolTable::setSymbolVisibility(function,
                                           SymbolTable::Visibility::Private);
        }
      }
      auto cloned = iluvatar_tle::raw::cloneLLVMSymbols(module, *parsed);
      if (failed(cloned)) {
        module.emitError("cannot import compiled raw source ") << sourceId;
        return signalPassFailure();
      }
      auto &symbols = imported[sourceId];
      for (const auto &item : *renamed)
        symbols[item.getKey()] = cloned->lookup(item.getValue());
      symbols[entry.externFuncName] = symbols.lookup(entrySymbol);
    }

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result =
        module.walk([&](iluvatar_tle::DSLRegionOp op) -> WalkResult {
          auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
          if (!sourceIdAttr || !op.getBody().empty())
            return WalkResult::advance();
          StringRef sourceId = sourceIdAttr.getValue();
          auto source = imported.find(sourceId);
          if (source == imported.end()) {
            op.emitError("missing imported raw source ") << sourceId;
            return WalkResult::interrupt();
          }
          StringRef originalName =
              compiled.find(sourceId)->second.externFuncName;
          auto function = dyn_cast_or_null<LLVM::LLVMFuncOp>(
              source->second.lookup(originalName));
          if (!function || function.isExternal()) {
            op.emitError("cannot find raw entry definition ")
                << originalName << " in source " << sourceId;
            return WalkResult::interrupt();
          }
          TritonOpBuilder builder(module.getContext());
          builder.setLastLoc(op.getLoc());
          if (failed(iluvatar_tle::raw::buildDSLRegionBodyFromLLVMFunc(
                  builder, op, function))) {
            op.emitError("cannot bind raw entry ")
                << originalName << " in source " << sourceId;
            return WalkResult::interrupt();
          }
          op->removeAttr(kSourceIdAttr);
          return WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  std::shared_ptr<const llvm::StringMap<CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<CompiledRaw>>();
};

std::unique_ptr<Pass>
createIluvatarMaterializeDeferredRaw(llvm::StringMap<CompiledRaw> sources) {
  return std::make_unique<IluvatarMaterializeDeferredRawPass>(
      std::move(sources));
}

} // namespace mlir
