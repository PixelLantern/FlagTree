#include "hcu/tle_raw/include/CompiledRaw.h"
#include "hcu/tle_raw/include/Passes.h"
#include "hcu/tle_raw/include/RawWrapperAdapter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/TleRawMaterialize.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <memory>

namespace tle = mlir::triton::tle;
namespace hcu_tle_raw = mlir::triton::HCU::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_HCUMATERIALIZEDEFERREDRAW
#include "hcu/tle_raw/include/Passes.h.inc"

class HCUMaterializeDeferredRawPass
    : public impl::HCUMaterializeDeferredRawBase<
          HCUMaterializeDeferredRawPass> {
public:
  using impl::HCUMaterializeDeferredRawBase<
      HCUMaterializeDeferredRawPass>::HCUMaterializeDeferredRawBase;

  HCUMaterializeDeferredRawPass() = default;
  explicit HCUMaterializeDeferredRawPass(
      llvm::StringMap<hcu_tle_raw::CompiledRaw> sources)
      : sources(
            std::make_shared<const llvm::StringMap<hcu_tle_raw::CompiledRaw>>(
                std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const auto &compiled = *sources;

    struct WrapperRequest {
      unsigned inputCount = 0;
      SmallVector<hcu_tle_raw::RawFragmentOutput> outputs;
    };
    llvm::StringMap<llvm::StringMap<WrapperRequest>> requests;
    module.walk([&](tle::DSLRegionOp op) {
      auto source = op->getAttrOfType<StringAttr>("tle_raw.source_id");
      if (!source || !op.getBody().empty())
        return;
      auto wrapper =
          op->getAttrOfType<FlatSymbolRefAttr>("tle_raw.wrapper_symbol");
      // Registration encodes the argument signature and output marks in the
      // wrapper symbol, so all calls with this symbol share one adapter.
      auto &bySymbol = requests[source.getValue()];
      if (bySymbol.count(wrapper.getValue()))
        return;
      auto &request = bySymbol[wrapper.getValue()];
      auto infos = op->getAttrOfType<ArrayAttr>("tle_raw.arg_info_list");
      for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
        auto kind = cast<StringAttr>(infos[i]);
        if (kind.getValue() == "tensor_fragment") {
          auto tensor = cast<RankedTensorType>(operand.getType());
          unsigned count = triton::gpu::getTotalElemsPerThread(tensor);
          if (llvm::is_contained(op.getOutputOperandIndices(), i))
            request.outputs.push_back({request.inputCount, count});
          request.inputCount += count;
        } else if (auto memdesc =
                       dyn_cast<triton::gpu::MemDescType>(operand.getType())) {
          request.inputCount += 3 + 2 * memdesc.getRank();
        } else {
          ++request.inputCount;
        }
      }
    });

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

      // Build before cloning: the imported module still has Clang's data
      // layout, including private allocation address space and alignments.
      llvm::StringMap<LLVM::LLVMFuncOp> entrypoints;
      for (const auto &item : requests[sourceId]) {
        auto function = parsed->lookupSymbol<LLVM::LLVMFuncOp>(item.getKey());
        if (!function) {
          module.emitError("cannot find raw wrapper ")
              << item.getKey() << " in source " << sourceId;
          return signalPassFailure();
        }
        const auto &request = item.getValue();
        auto adapter = hcu_tle_raw::createRawWrapperAdapter(
            *parsed, function, request.inputCount, request.outputs);
        if (failed(adapter))
          return signalPassFailure();
        entrypoints[item.getKey()] = *adapter;
      }

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
      for (auto &item : entrypoints)
        symbols[item.getKey()] = cloned->lookup(item.getValue().getOperation());
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
  std::shared_ptr<const llvm::StringMap<hcu_tle_raw::CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<hcu_tle_raw::CompiledRaw>>();
};

std::unique_ptr<Pass> createHCUMaterializeDeferredRaw(
    llvm::StringMap<hcu_tle_raw::CompiledRaw> sources) {
  return std::make_unique<HCUMaterializeDeferredRawPass>(std::move(sources));
}

} // namespace mlir
